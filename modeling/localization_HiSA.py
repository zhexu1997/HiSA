import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from modeling.dynamic_filters.build import DynamicFilter, HISA_query, HISA_video, Video_Dis, Conv1D
from utils import loss as L
from utils.rnns import feed_forward_rnn
import utils.pooling as POOLING
from modeling.trm import Encoder


class Localization_HiSA(nn.Module):
    def __init__(self, cfg):
        super(Localization_HiSA, self).__init__()
        self.cfg = cfg
        self.batch_size = cfg.BATCH_SIZE_TRAIN
        # self.model_df = ACRM_query(cfg)
        self.model_df = HISA_query(cfg)

        self.model_video_GRU = HISA_video(cfg)
        self.multimodal_fc1 = nn.Linear(512 * 2, 1)
        self.multimodal_fc2 = nn.Linear(512, 1)
        self.is_use_rnn_loc = cfg.HISA_CLASSIFICATION.USED
        self.rnn_localization = nn.LSTM(input_size=cfg.HISA_CLASSIFICATION.INPUT_SIZE,
                                        hidden_size=cfg.HISA_CLASSIFICATION.INPUT_SIZE_RNN,
                                        num_layers=cfg.LOCALIZATION.HISA_NUM_LAYERS,
                                        bias=cfg.LOCALIZATION.BIAS,
                                        dropout=cfg.LOCALIZATION.DROPOUT,
                                        bidirectional=cfg.LOCALIZATION.BIDIRECTIONAL,
                                        batch_first=cfg.LOCALIZATION.BATCH_FIRST)

        if cfg.HISA_CLASSIFICATION.FUSION == 'CAT':
            cfg.HISA_CLASSIFICATION.INPUT_SIZE = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (
                        1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL)) \
                                                 + cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE * (
                                                             1 + int(cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL))
        else:
            assert cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL)) == \
                   cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL))
            cfg.HISA_CLASSIFICATION.INPUT_SIZE = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (
                        1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL))
        if cfg.HISA_CLASSIFICATION.USED:
            cfg.HISA_CLASSIFICATION.INPUT_SIZE = cfg.HISA_CLASSIFICATION.INPUT_SIZE_RNN * (
                        1 + int(cfg.LOCALIZATION.BIDIRECTIONAL))

        self.pooling = POOLING.MeanPoolingLayer()
        self.starting = nn.Sequential(
            nn.Linear(cfg.HISA_CLASSIFICATION.INPUT_SIZE, cfg.HISA_CLASSIFICATION.HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(cfg.HISA_CLASSIFICATION.HIDDEN_SIZE, cfg.HISA_CLASSIFICATION.OUTPUT_SIZE))
        self.ending = nn.Sequential(
            nn.Linear(cfg.HISA_CLASSIFICATION.INPUT_SIZE, cfg.HISA_CLASSIFICATION.HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(cfg.HISA_CLASSIFICATION.HIDDEN_SIZE, cfg.HISA_CLASSIFICATION.OUTPUT_SIZE))
        self. intering = nn.Sequential(
            nn.Linear(cfg.HISA_CLASSIFICATION.INPUT_SIZE, cfg.HISA_CLASSIFICATION.HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(cfg.HISA_CLASSIFICATION.HIDDEN_SIZE, cfg.HISA_CLASSIFICATION.OUTPUT_SIZE))
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        self.video_dis = Video_Dis(cfg)
        self.trm_fused = Encoder(d_model=512, n=1, heads=8)

    def get_mask_from_sequence_lengths(self, sequence_lengths: torch.Tensor, max_length: int):
        ones = sequence_lengths.new_ones(sequence_lengths.size(0), max_length)
        range_tensor = ones.cumsum(dim=1)
        return (sequence_lengths.unsqueeze(1) >= range_tensor).long()

    def fusion_layer(self, filter_start, output_video, mode):
        if mode == 'CAT':
            output = torch.cat([filter_start.unsqueeze(dim=1).repeat(1, output_video.shape[1], 1), output_video],
                               dim=-1)
        elif mode == 'COS':
            output = filter_start.unsqueeze(dim=1).repeat(1, output_video.shape[1], 1) * output_video
        elif mode == 'SUB':
            output = (filter_start.unsqueeze(dim=1).repeat(1, output_video.shape[1], 1) - output_video)
        elif mode == 'CROSS_COS':
            output = filter_start * output_video
        elif mode == 'CROSS_SUB':
            output = torch.abs(filter_start - output_video)
        return output

    def masked_softmax(self, vector: torch.Tensor, mask: torch.Tensor, dim: int = -1, memory_efficient: bool = False,
                       mask_fill_value: float = -1e32):
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)

        return result + 1e-13

    def mask_softmax(self, feat, mask):
        return self.masked_softmax(feat, mask, memory_efficient=False)

    def max_boundary(self, p, gt, length):
        individual_loss = []
        for i in range(length.size(0)):
            # vlength = int(length[i])
            index_bd = gt[i]
            ret = torch.log(p[i][index_bd])
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def max_inter(self, p, gt_s, gt_e, length):
        individual_loss = []
        for i in range(length.size(0)):
            vlength = int(length[i])
            index_bs = gt_s[i]
            index_be = gt_e[i]
            ret = torch.log(p[i][index_bs:(index_be + 1)]) / (max(index_be - index_bs, 1))
            individual_loss.append(-torch.sum(ret))
        individual_loss = torch.stack(individual_loss)
        return torch.mean(individual_loss), individual_loss

    def compute_contrast_loss(self, text, video, frame_start, frame_end, mask, weighting=True, pooling='mean', tao=0.2):
        b, _, d = text.shape
        text_global = torch.ones(b, d).cuda()
        video_global = torch.ones(b, d).cuda()
        for i in range(b):
            if pooling == 'mean':
                text_global[i] = torch.sum(text[i][frame_start[i]:frame_end[i]]) / (frame_end[i] - frame_start[i])
            elif pooling == 'max':
                text_global[i] = torch.max(text[i][frame_start[i]:frame_end[i]], 0)[0]
            video_global[i] = torch.sum(video[i][frame_start[i]:frame_end[i]]) / (frame_end[i] - frame_start[i])

        vcon_loss = 0
        tcon_loss = 0
        if weighting:
            for i in range(b):
                weighting = 1 - nn.CosineSimilarity(dim=1)(text_global[i].expand(text_global.size()), text_global)
                weighting[i] = 1
                cos_similarity = nn.CosineSimilarity(dim=1)(text_global[i].expand(text_global.size()), video_global)
                cos_similarity = torch.exp(cos_similarity/tao) * weighting
                vcon_loss += (-1) * torch.log(cos_similarity[i] / cos_similarity.sum())
            for i in range(b):
                weighting = 1 - nn.CosineSimilarity(dim=1)(video_global[i].expand(video_global.size()), video_global)
                weighting[i] = 1
                cos_similarity = nn.CosineSimilarity(dim=1)(video_global[i].expand(video_global.size()), text_global)
                cos_similarity = torch.exp(cos_similarity/tao) * weighting
                tcon_loss += (-1) * torch.log(cos_similarity[i] / cos_similarity.sum())
        else:
            for i in range(b):
                cos_similarity = nn.CosineSimilarity(dim=1)(text_global[i].expand(text_global.size()), video_global)
                cos_similarity = torch.exp(cos_similarity)
                vcon_loss += (-1) * torch.log(cos_similarity[i] / cos_similarity.sum())
            for i in range(b):
                cos_similarity = nn.CosineSimilarity(dim=1)(video_global[i].expand(video_global.size()), text_global)
                cos_similarity = torch.exp(cos_similarity)
                tcon_loss += (-1) * torch.log(cos_similarity[i] / cos_similarity.sum())

        con_loss = (vcon_loss + tcon_loss) / b
        return con_loss

    def forward(self, videoFeat, videoFeat_lengths, tokens, tokens_lengths, start, end, localiz, frame_start,
                frame_end):
        videoFeat, loss_re = self.video_dis(videoFeat)
        mask = self.get_mask_from_sequence_lengths(videoFeat_lengths, int(videoFeat.shape[1]))
        output_video = self.model_video_GRU(videoFeat, videoFeat_lengths, mask)
        filter_start, lengths, hncn = self.model_df(tokens, tokens_lengths, output_video)
        output = self.fusion_layer(filter_start, output_video, self.cfg.HISA_CLASSIFICATION.FUSION)
        if self.is_use_rnn_loc:
            output, _ = feed_forward_rnn(self.rnn_localization, output, lengths=videoFeat_lengths)
        # output = self.trm_fused(output)

        pred_start = self.starting(output.view(-1, output.size(2))).view(-1, output.size(1), 1).squeeze()
        pred_start = self.mask_softmax(pred_start, mask)
        pred_end = self.ending(output.view(-1, output.size(2))).view(-1, output.size(1), 1).squeeze()
        pred_end = self.mask_softmax(pred_end, mask)
        pred_inter = self.intering(output.view(-1, output.size(2))).view(-1, output.size(1), 1).squeeze()
        pred_inter = self.mask_softmax(pred_inter, mask)

        start_loss, _ = self.max_boundary(pred_start, frame_start, videoFeat_lengths)
        end_loss, _ = self.max_boundary(pred_end, frame_end, videoFeat_lengths)
        inter_loss, _ = self.max_inter(pred_inter, frame_start, frame_end, videoFeat_lengths)
        inter_loss *= 1
        total_loss = start_loss + end_loss + inter_loss

        ccloss = self.compute_contrast_loss(filter_start, output, frame_start, frame_end, mask, weighting=True, pooling='mean', tao=1)
        ccloss *= 0.5
        loss_re *= 50
        total_loss += loss_re
        total_loss += ccloss
        print('start loss:{}, end_loss:{}, inter_loss:{}, re_loss:{}, con_loss:{}, total_loss:{}'.format(start_loss, end_loss, inter_loss, loss_re, ccloss, total_loss))
        return total_loss, pred_start, pred_end


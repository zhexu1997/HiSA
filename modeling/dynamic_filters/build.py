import torch
import numpy as np
from torch import nn
import modeling.dynamic_filters as DF
import utils.pooling as POOLING


class DynamicFilter(nn.Module):
    def __init__(self, cfg):
        super(DynamicFilter, self).__init__()
        self.cfg = cfg
        factory = getattr(DF, cfg.DYNAMIC_FILTER.TAIL_MODEL)
        self.tail_df = factory(cfg)
        factory = getattr(POOLING, cfg.DYNAMIC_FILTER.POOLING)
        self.pooling_layer = factory()
        factory = getattr(DF, cfg.DYNAMIC_FILTER.HEAD_MODEL)
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths=None, video_fea=None):
        output, _ = self.tail_df(sequences, lengths)
        output = self.pooling_layer(output, video_fea=None, lengths=lengths)
        output = self.head_df(output)
        return output, lengths 


# class ACRM_query(nn.Module):


class HISA_query(nn.Module):
    def __init__(self, cfg):
        super(HISA_query, self).__init__()
        self.cfg = cfg
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        factory = getattr(DF, cfg.HISA_QUERY.TAIL_MODEL)
        self.tail_df = factory(cfg)
        factory = getattr(POOLING, cfg.HISA_QUERY.POOLING)
        self.pooling_layer = factory()
        factory = getattr(DF, cfg.HISA_QUERY.HEAD_MODEL)
        self.head_df = factory(cfg)
        self.query_hidden =cfg.DYNAMIC_FILTER.LSTM.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM.BIDIRECTIONAL))
        self.video_hidden = cfg.DYNAMIC_FILTER.LSTM_VIDEO.HIDDEN_SIZE * (1 + int(cfg.DYNAMIC_FILTER.LSTM_VIDEO.BIDIRECTIONAL))
        if cfg.HISA_CLASSIFICATION.FUSION == 'CROSS_COS' or cfg.HISA_CLASSIFICATION.FUSION == 'CROSS_SUB':
            self.pooling_layer = DF.InteractorwoLSTM(self.query_hidden, self.video_hidden, self.query_hidden)

    def forward(self, sequences, lengths=None, video_fea=None):
        output, hncn = self.tail_df(sequences, lengths)
        output = self.dropout_layer(output)
        output = self.pooling_layer(output, video_fea, lengths)
        output = self.head_df(output)
        return output, lengths, hncn


class HISA_video(nn.Module):
    def __init__(self, cfg):
        super(HISA_video, self).__init__()
        self.cfg = cfg
        self.dropout_layer = nn.Dropout(cfg.DYNAMIC_FILTER.LSTM_VIDEO.DROPOUT)
        factory = getattr(DF, cfg.HISA_VIDEO.TAIL_MODEL)
        self.tail_df = factory(cfg)
        factory = getattr(DF, cfg.HISA_VIDEO.HEAD_MODEL)
        self.head_df = factory(cfg)

    def forward(self, sequences, lengths, masks = None):
        output, _ = self.tail_df(sequences, lengths, masks)
        output = self.dropout_layer(output)
        # output = self.pooling_layer(output, lengths)
        output = self.head_df(output)
        return output


class Conv1D(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=True):
        super(Conv1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_dim, out_channels=out_dim, kernel_size=kernel_size, padding=padding,
                                stride=stride, bias=bias)

    def forward(self, x):
        # suppose all the input with shape (batch_size, seq_len, dim)
        x = x.transpose(1, 2)  # (batch_size, dim, seq_len)
        x = self.conv1d(x)
        return x.transpose(1, 2)  # (batch_size, seq_len, dim)


class VisualProjection(nn.Module):
    """conv1d"""
    def __init__(self, visual_dim, dim, drop_rate=0.0, k=1, p=0):
        super(VisualProjection, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = Conv1D(in_dim=visual_dim, out_dim=dim, kernel_size=k, stride=1, bias=True, padding=p)

    def forward(self, visual_features):
        # the input visual feature with shape (batch_size, seq_len, visual_dim)
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output


class VisualProjection1(nn.Module):
    """linear"""
    def __init__(self, visual_dim, hidden_dim, dim, drop_rate=0.0):
        super(VisualProjection1, self).__init__()
        self.drop = nn.Dropout(p=drop_rate)
        self.linear = nn.Sequential(nn.Linear(visual_dim, hidden_dim), nn.Tanh())

    def forward(self, visual_features):
        visual_features = self.drop(visual_features)
        output = self.linear(visual_features)  # (batch_size, seq_len, dim)
        return output


class Video_Dis(nn.Module):
    def __init__(self, cfg):
        super(Video_Dis, self).__init__()
        self.cfg = cfg
        self.background_encoder = VisualProjection(visual_dim=1024 * 2, dim=1024, drop_rate=0.5, k=5, p=2)
        self.action_encoder = VisualProjection(visual_dim=1024 * 2, dim=1024, drop_rate=0.5, k=5, p=2)
        self.video_decoder = VisualProjection(visual_dim=1024 * 2, dim=1024, drop_rate=0.5, k=5, p=2)

    def forward(self, video_features):
        b, t, _ = video_features.shape
        perm_new = torch.cat([torch.arange(1, t), torch.arange(t - 1, t)], dim=0)
        neighbor_features = video_features[:, perm_new, :]
        video_background_features = self.background_encoder(torch.cat((video_features, neighbor_features), dim=2))
        video_action_features = self.action_encoder(torch.cat((video_features, video_background_features), dim=2))
        video_action_features_neighbor = self.action_encoder(torch.cat((neighbor_features, video_background_features), dim=2))
        reconstructed_video_features = self.video_decoder(torch.cat((video_action_features, video_background_features), dim=2))
        reconstructed_video_features_neighbor = self.video_decoder(torch.cat((video_action_features_neighbor, video_background_features), dim=2))
        loss_re = nn.L1Loss()(video_features, reconstructed_video_features)
        loss_re += nn.L1Loss()(neighbor_features, reconstructed_video_features_neighbor)
        video_features = torch.cat((video_action_features, video_background_features), dim=2)
        return video_features, loss_re

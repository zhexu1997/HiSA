import os
import json
import time
import math
import torch
import pickle
import numpy as np
from random import shuffle
import torchtext
import h5py
import torch.nn as nn
from utils.vocab import Vocab
from utils.sentence import get_embedding_matrix

from torch.utils.data import Dataset

from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


class ANET_CAP(Dataset):
    vocab = torchtext.vocab.pretrained_aliases['glove.840B.300d'](cache='/home/xz/Projects/HiSA/data')
    vocab.itos.extend(['<unk>'])
    vocab.stoi['<unk>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)

    def __init__(self, features_path,
                       ann_file_path,
                       embeddings_path,
                       min_count,
                       train_max_length,
                       test_max_length):

        self.feature_path = features_path
        self.ann_file_path = ann_file_path
        self.is_training = 'training' in ann_file_path
        self.sample_rate = 1
        print(self.is_training)
        print('loading anns into memory...', end=" ")
        tic = time.time()
        # self.dataset = json.load(open(ann_file_path, 'r'))
        # self.glove = np.load(vocab_glove, allow_pickle=True).item()
        print('Done (t={:0.2f}s)'.format(time.time()- tic))
        self.min_count = min_count
        self.train_max_length = train_max_length
        self.test_max_length = test_max_length
        self.dataset = json.load(open(ann_file_path, 'r'))
        visual_features = h5py.File(self.feature_path, 'r')
        existed_fea = list(visual_features.keys())

        anno_pairs = []
        for vid, video_anno in self.dataset.items():
            duration = video_anno['duration']
            if vid not in existed_fea:
                continue
            for timestamp, sentence in zip(video_anno['timestamps'], video_anno['sentences']):
                if timestamp[0] < timestamp[1]:
                    anno_pairs.append(
                        {
                            'video': vid,
                            'duration': duration,
                            'times':[max(timestamp[0],0),min(timestamp[1],duration)],
                            'description':sentence,
                        }
                    )
        self.anns = anno_pairs

        self.ids = [i['video'] for i in self.anns]
        self.epsilon = 1E-10

    def createIndex(self):
        print("Creating index..", end=" ")
        anns = {}
        size = int(round(len(self.dataset) * 1.))
        counter = 0
        for row in self.dataset[:size]:
            if self.is_training:
                if float(row['number_features']) < 10:
                    continue            # print(row) 
                if float(row['number_features']) >= 1200:
                    continue            # print(row)
            if float(row['feature_start']) > float(row['feature_end']):
                continue
            if math.floor(float(row['feature_end'])) >= float(row['number_features']):
                row['feature_end'] = float(row['number_features'])-1
            if self.is_training:
                row['augmentation'] = 1
                anns[counter] = row.copy()
                counter += 1

            row['augmentation'] = 0
            anns[counter] = row
            counter+=1
        self.anns = anns
        print(" Ok! {}".format(len(anns.keys())))

    def __getitem__(self, index):
        ann = self.anns[index]
        # print(ann)
        video_id = self.anns[index]['video']
        time_start, time_end = self.anns[index]['times']
        sentence = self.anns[index]['description']
        duration = float(self.anns[index]['duration'])

        visual_features = h5py.File(self.feature_path, 'r')
        i3dfeat = torch.from_numpy(visual_features[ann['video']]['c3d_features'][:]).float()

        i3dfeat =  i3dfeat[list(range(0, i3dfeat.shape[0], self.sample_rate))]
        feat_length = i3dfeat.shape[0]

        word_idxs = torch.tensor([self.vocab.stoi.get(w.lower(), 10000) for w in sentence.split()], dtype=torch.long)
        tokens = self.word_embedding(word_idxs)

        feature_start = int(feat_length* time_start / duration)
        feature_end = min(math.ceil(feat_length*time_end/duration),feat_length-1)
        fps = 30
        factor = 8 * self.sample_rate
        if False:
            # feature_start = ann['feature_start']
            # feature_end   = ann['feature_end']

            offset = int(math.floor(feature_start))
            if offset != 0:
                offset = np.random.randint(0, int(round(feature_start)))

            new_feature_start = feature_start - offset
            new_feature_end   = feature_end - offset

            i3dfeat = i3dfeat[offset:,:]

            duration_new = (feat_length - offset)* duration /feat_length
            feat_length = feat_length - offset

            localization = np.zeros(feat_length, dtype=np.float32)

            start = math.floor(new_feature_start)
            end   = math.floor(new_feature_end)

            time_start = new_feature_start *  duration_new / (feat_length)
            # time_start = (new_feature_start *factor) / fps
            # time_end = (new_feature_end * factor) / fps
            time_offset = (offset * factor) / fps


            time_end = min(new_feature_end * duration_new / (feat_length),duration_new)

        else:
            localization = np.zeros(feat_length, dtype=np.float32)

            # loc_start =
            start = feature_start
            end = feature_end

        loc_start = np.ones(feat_length, dtype=np.float32) * self.epsilon
        loc_end = np.ones(feat_length, dtype=np.float32) * self.epsilon
        y = (1 - (feat_length-3) * self.epsilon - 0.5)/ 2
        # print(y)
        if start > 0:
            loc_start[start - 1] = y
        if start < feat_length-1:
            loc_start[start + 1] = y
        loc_start[start] = 0.5

        if end > 0:
            loc_end[end - 1] = y
        if end < feat_length-1:
            loc_end[end + 1] = y
        loc_end[end] = 0.5

        y = 1.0
        localization[start:end] = y

        return index, i3dfeat, tokens, torch.from_numpy(loc_start), torch.from_numpy(loc_end), torch.from_numpy(localization),\
                   time_start, time_end, factor, fps, start, end, duration, video_id

    def __len__(self):
        return len(self.anns)

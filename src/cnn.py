import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
# Adapted from https://www.kaggle.com/mlwhiz/textcnn-pytorch-and-keras


class CNNClassifier(nn.Module):

    def __init__(self, window_sizes: List[int], no_filters: int, max_feats: int, embedding_dim: int, no_classes: int):
        """Initialise the model.
        :param window_sizes: The size of the filters (e.g. 1: unigram, 2: bigram, etc.)
        :param no_filters: The number of filters to apply.
        :param max_feats: The maximum length of the sequence to consider.
        """
        self.embedding = nn.Embedding(max_feats, embedding_dim)
        self.conv = [nn.ModuleList(nn.Conv2d(1, no_filters, (w, embedding_dim)) for w in window_sizes)]
        self.linear = nn.Linear(len(window_sizes) * no_filters, no_classes)

    def forward(self, sequence):
        """The forward step of the model.
        :param sequence: The sequence to be predicted on.
        :return scores: The scores computed by the model.
        """
        emb = self.embedding(sequence)
        emb = emb.unsqueeze(1)
        output = [F.relu(conv(sequence)).squeeze(3) for conv in self.conv]
        output = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in sequence]
        output = torch.cat(output, 1)
        scores = self.linear(output)

        return scores


cnn = CNNClassifier([1, 2, 3, 4], 128, 200, 300)

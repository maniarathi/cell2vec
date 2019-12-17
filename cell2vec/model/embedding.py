import math

import torch


class Embedding(torch.nn.Module):
    def __init__(self, model_dimensionality, vocab):
        super(Embedding, self).__init__()

        self.embedding_matrix = torch.nn.Embedding(vocab, model_dimensionality)
        self.model_dimensionality = model_dimensionality

    def forward(self, X):
        return self.embedding_matrix(X) * math.sqrt(self.model_dimensionality)

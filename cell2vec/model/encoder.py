import torch

from copy import deepcopy
from .encoder_layer import EncoderLayer
from .layer_norm import LayerNorm


class Encoder(torch.nn.Module):
    def __init__(self, model_size, number_of_layers, encoder_layer):
        super(Encoder, self).__init__()

        self.encoder = torch.nn.ModuleList([deepcopy(encoder_layer) for _ in range(number_of_layers)])
        self.layer_norm = torch.nn.LayerNorm(model_size, eps=1e-06)


    def forward(self, X):
        for encoder in self.encoder:
            X = encoder(X)

        return self.layer_norm(X)

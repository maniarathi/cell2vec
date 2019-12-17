import torch

from .encoder_layer import EncoderLayer
from .layer_norm import LayerNorm


class Decoder(torch.nn.Module):
    def __init__(self, model_size, number_of_layers, decoder_layer):
        super(Decoder, self).__init__()

        self.decoder = torch.nn.ModuleList([decoder_layer for _ in range(number_of_layers)])
        self.layer_norm = torch.nn.LayerNorm(model_size, eps=1e-06)


    def forward(self, X, encoder_memory):
        for decoder in self.decoder:
            X = decoder(X, encoder_memory)

        return self.layer_norm(X)

from copy import deepcopy

import torch

from .decoder import Decoder
from .decoder_layer import DecoderLayer
from .embedding import Embedding
from .encoder import Encoder
from .encoder_decoder import EncoderDecoder
from .encoder_layer import EncoderLayer
from .feed_forward import FeedForward
from .generator import Generator
from .multi_head_attention import MultiHeadAttention


class TransformerModel:
    def __init__(self, source_vocab, target_vocab, number_of_stacks=6, model_dimensionality=512,
                 feed_forward_dimensionality=2048, number_of_attention_heads=8, dropout=0.1):
        attention = MultiHeadAttention(model_dimensionality, number_of_attention_heads)
        feed_forward = FeedForward(model_dimensionality, feed_forward_dimensionality, dropout)

        encoder_model = Encoder(model_dimensionality, number_of_stacks,
                                EncoderLayer(deepcopy(attention), deepcopy(feed_forward), model_dimensionality))
        decoder_model = Decoder(model_dimensionality, number_of_stacks,
                                DecoderLayer(deepcopy(attention), deepcopy(attention), deepcopy(feed_forward),
                                             model_dimensionality))

        input_embeddings = Embedding(model_dimensionality, source_vocab)
        output_embeddings = Embedding(model_dimensionality, target_vocab)

        generator = Generator(model_dimensionality, target_vocab)

        self.model = EncoderDecoder(encoder_model, decoder_model, input_embeddings, output_embeddings,
                                    generator)

        for param in self.model.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform(param)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def forward(self, inputs, outputs):
        self.model.forward(inputs, outputs)
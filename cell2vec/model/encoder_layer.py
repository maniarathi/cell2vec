import torch

from .layer_norm import LayerNorm


class EncoderLayer(torch.nn.Module):
    def __init__(self, multi_head_attention, feed_forward, model_size):
        super(EncoderLayer, self).__init__()

        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward

        self.layer_norm_after_attention = LayerNorm(model_size)
        self.layer_norm_after_feed_forward = LayerNorm(model_size)

    def forward(self, X):
        attention_output = self.layer_norm_after_attention(X, self.multi_head_attention(X, X, X))
        return self.layer_norm_after_feed_forward(attention_output, self.feed_forward(attention_output))

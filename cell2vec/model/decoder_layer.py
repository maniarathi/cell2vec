import torch

from .layer_norm import LayerNorm


class DecoderLayer(torch.nn.Module):
    def __init__(self, masked_multi_head_attention, multi_head_attention, feed_forward, model_size):
        super(DecoderLayer, self).__init__()

        self.masked_multi_head_attention = masked_multi_head_attention
        self.multi_head_attention = multi_head_attention
        self.feed_forward = feed_forward

        self.layer_norm_after_masked_attention = LayerNorm(model_size)
        self.layer_norm_after_attention = LayerNorm(model_size)
        self.layer_norm_after_feed_forward = LayerNorm(model_size)

    def forward(self, X, encoder_memory):
        masked_attention_output = self.layer_norm_after_masked_attention(X, self.masked_multi_head_attention(X, X, X))
        attention_output = self.layer_norm_after_attention(masked_attention_output,
                                                           self.multi_head_attention(encoder_memory, encoder_memory,
                                                                                     masked_attention_output))
        return self.layer_norm_after_feed_forward(attention_output, self.feed_forward(attention_output))

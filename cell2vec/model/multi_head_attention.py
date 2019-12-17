import math

import torch


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, model_size, number_of_attention_heads):
        super(MultiHeadAttention, self).__init__()

        self.number_of_attention_heads = number_of_attention_heads
        self.key_dimensions = model_size // number_of_attention_heads

        self.linear_v = torch.nn.Linear(model_size, model_size)
        self.linear_k = torch.nn.Linear(model_size, model_size)
        self.linear_q = torch.nn.Linear(model_size, model_size)

        self.linear_final = torch.nn.Linear(model_size, model_size)

    def forward(self, query, key, value):
        num_batches = query.size(0)

        linearified_query = self.linear_q(query)
        linearified_key = self.linear_k(key)
        linearified_value = self.linear_v(value)

        shaped_query = linearified_query.view(num_batches, -1, self.number_of_attention_heads,
                                              self.key_dimensions).transpose(1, 2)
        shaped_key = linearified_key.view(num_batches, -1, self.number_of_attention_heads,
                                          self.key_dimensions).transpose(1, 2)
        shaped_value = linearified_value.view(num_batches, -1, self.number_of_attention_heads,
                                              self.key_dimensions).transpose(1, 2)

        attentionified = self.scaled_dot_product_attention(shaped_query, shaped_key, shaped_value)

        attentionified = attentionified.transpose(1, 2).contiguous().view(num_batches, -1,
                                                                          self.number_of_attention_heads *
                                                                          self.key_dimensions)

        return self.linear_final(attentionified)

    def scaled_dot_product_attention(self, query_input, key_input, value_input):
        multiplied_query_key_t = torch.matmul(query_input, torch.transpose(key_input, -2, -1))
        scaled = multiplied_query_key_t / math.sqrt(key_input.size(-1))
        return torch.matmul(torch.nn.Softmax(scaled, dim=-1), value_input)

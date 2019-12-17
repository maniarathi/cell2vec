import torch


class LayerNorm(torch.nn.Module):
    def __init__(self, model_size, dropout_rate=0.1, layer_norm_epsilon=1e-06):
        super(LayerNorm, self).__init__()

        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(model_size, eps=layer_norm_epsilon)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))

import torch

class FeedForward(torch.nn.Module):
    def __init__(self, input_and_output_size, hidden_size, dropout_rate=0.1):
        super(FeedForward, self).__init__()

        self.w1 = torch.nn.Linear(input_and_output_size, hidden_size)
        self.w2 = torch.nn.Linear(hidden_size, input_and_output_size)
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        return self.w2(self.dropout(torch.nn.ReLU(self.w1(x))))
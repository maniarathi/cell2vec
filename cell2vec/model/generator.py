import torch

class Generator(torch.nn.Module):
    def __init__(self, model_dimensionality, vocab):
        super(Generator, self).__init__()

        self.output_probabilities = torch.nn.Linear(model_dimensionality, vocab)

    def forward(self, X):
        return torch.nn.functional.log_softmax(self.output_probabilities(X), dim=-1)
import torch


class EncoderDecoder(torch.nn.Module):
    def __init__(self, encoder, decoder, input_embedding, output_embedding, generator):
        super(EncoderDecoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.generator = generator

    def forward(self, inputs, outputs):
        encoded_inputs = self.encoder(self.input_embedding(inputs))
        return self.decoder(self.output_embedding(outputs), encoded_inputs)

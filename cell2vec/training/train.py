import time

import anndata
import torch

from cell2vec.model.transformer_model import TransformerModel
from .optimizer import TransformerOptimizer

global max_src_in_batch, max_tgt_in_batch


class Train:

    def __init__(self, data_file, train_split=.8, num_epoch=10):
        self.num_epoch = num_epoch

        self.train_data, self.test_data = self.get_train_test_data(data_file, train_split)

        self.model = TransformerModel()
        self.optimizer = TransformerOptimizer(
            torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0,
                             amsgrad=False), 512)

    def train(self):
        for i in range(self.num_epoch):
            self.model.train()
            self.run_epoch()
            self.model.eval()

    def run_epoch(self, data_iter):
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            out = self.model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
            loss = self.calculate_loss(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 50 == 1:
                elapsed = time.time() - start
                print(f"Epoch Step: {i} Loss: {loss / batch.ntokens} Tokens per Sec: {tokens / elapsed}")
                start = time.time()
                tokens = 0
        return total_loss / total_tokens

    def calculate_loss(self):
        pass

    def create_data_iterator(self):
        pass

    def batch_size_fn(new, count, sofar):
        "Keep augmenting batch and calculate total number of tokens + padding."
        global max_src_in_batch, max_tgt_in_batch
        if count == 1:
            max_src_in_batch = 0
            max_tgt_in_batch = 0
        max_src_in_batch = max(max_src_in_batch, len(new.src))
        max_tgt_in_batch = max(max_tgt_in_batch, len(new.trg) + 2)
        src_elements = count * max_src_in_batch
        tgt_elements = count * max_tgt_in_batch
        return max(src_elements, tgt_elements)

    def get_train_test_data(self, data_file, split):
        file_object = anndata.read_h5ad(data_file)
        matrix = file_object.X

        num_cells_in_training = int(matrix.shape[1] * split)

        transpose = matrix.transpose()

        training_data = transpose[0:num_cells_in_training][:].transpose()
        test_data = transpose[num_cells_in_training:][:].transpose()

        print(f"Training data size : {training_data.shape}. Test data size : {test_data.shape}.")
        return training_data, test_data

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import AliasMultinomial


class SGNSLoss(nn.Module):
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 15  # Taken from Moody's OG code

    def __init__(self, dataset, word_embeddings, device):
        super(SGNSLoss, self).__init__()
        self.dataset = dataset
        self.vocab_len = word_embeddings.weight.size()[0]
        self.word_embeddings = word_embeddings
        self.device = device

        # Helpful values for unigram distribution generation
        self.transformed_freq_vec = t.tensor(
            np.array(list(dataset.term_freq_dict.values())) ** self.BETA
        )
        self.freq_sum = t.sum(self.transformed_freq_vec)

        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, center, context):
        center, context = center.squeeze(), context.squeeze()  # batch_size x embed_size

        # Compute non-sampled portion
        dots = (center * context).sum(-1)  # batch_size
        log_targets = F.logsigmoid(dots)

        # Compute sampled portion
        samples = self.get_unigram_samples()  # num_samples x batch_size
        log_samples = []
        for s in samples:
            dot = (t.neg(center) * s).sum(-1)  # batch_size
            log_samples.append(F.logsigmoid(dot))

        log_samples = t.stack(log_samples).sum(0)
        return t.add(log_targets, log_samples).mean().neg()  # Negative so goes towards loss of 0

    def get_unigram_samples(self, N=NUM_SAMPLES):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        rand_idxs = self.unigram_table.draw(N).to(self.device)
        return self.word_embeddings(rand_idxs).squeeze()

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        PDF = []
        for token_idx in range(0, self.vocab_len):
            PDF.append(self.get_unigram_prob(token_idx))
        # Generate the table from PDF
        return AliasMultinomial(PDF, self.device)
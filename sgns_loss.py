import torch as t
import torch.nn as nn
import numpy as np
from utils import AliasMultinomial


class SGNSLoss(nn.Module):
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 2

    def __init__(self, dataset, word_embeddings, device):
        super(SGNSLoss, self).__init__()
        self.dataset = dataset
        self.criterion = nn.BCEWithLogitsLoss()
        self.vocab_len = len(dataset.dictionary)
        self.word_embeddings = word_embeddings
        self.device = device

        # Helpful values for unigram distribution generation
        # Should use cfs instead but: https://github.com/RaRe-Technologies/gensim/issues/2574
        self.transformed_freq_vec = t.tensor(
            np.array([dataset.dictionary.dfs[i] for i in range(self.vocab_len)]) ** self.BETA
        )
        self.freq_sum = t.sum(self.transformed_freq_vec)
        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, center, context):
        center, context = center.squeeze(), context.squeeze()  # batch_size x embed_size

        # Compute true portion
        true_scores = (center * context).sum(-1)  # batch_size
        loss = self.criterion(true_scores, t.ones_like(true_scores))
        #test_loss = loss.detach().item()

        # Compute negatively sampled portion -
        for i in range(self.NUM_SAMPLES):
            samples = self.get_unigram_samples(n=center.shape[0])
            neg_sample_scores = (center * samples).sum(-1)
            # Update loss
            loss += self.criterion(neg_sample_scores, t.zeros_like(neg_sample_scores))

            #x3 = neg_sample_scores.clone().detach().numpy()
            #test_loss += self.bce_loss_w_logits(x3, t.zeros_like(neg_sample_scores).numpy())

        return loss#, test_loss

    @staticmethod
    def bce_loss_w_logits(x, y):
        max_val = np.clip(x, 0, None)
        loss = x - x * y + max_val + np.log(np.exp(-max_val) + np.exp((-x - max_val)))
        return loss.mean()

    def get_unigram_samples(self, n):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        rand_idxs = self.unigram_table.draw(n).to(self.device)
        return self.word_embeddings(rand_idxs).squeeze()

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        # Probability at each index corresponds to probability of selecting that token
        pdf = [self.get_unigram_prob(t_idx) for t_idx in range(0, self.vocab_len)]
        # Generate the table from PDF
        return AliasMultinomial(pdf, self.device)

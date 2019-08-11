import torch as t
import numpy as np


class AliasMultinomial(object):
    """
    Fast sampling from a multinomial distribution.
    https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    Code taken from: https://github.com/TropComplique/lda2vec-pytorch/blob/master/utils/alias_multinomial.py
    """

    def __init__(self, probs, device):
        """
        probs: a float tensor with shape [K].
            It represents probabilities of different outcomes.
            There are K outcomes. Probabilities sum to one.
        """
        self.device = device

        K = len(probs)
        self.q = t.zeros(K).to(device)
        self.J = t.LongTensor([0] * K).to(device)

        # sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            self.q[kk] = K * prob
            if self.q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            self.J[small] = large
            self.q[large] = (self.q[large] - 1.0) + self.q[small]

            if self.q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.q.clamp(0.0, 1.0)
        self.J.clamp(0, K - 1)

    def draw(self, N):
        """Draw N samples from the distribution."""

        K = self.J.size(0)
        r = t.LongTensor(np.random.randint(0, K, size=N)).to(self.device)
        q = self.q.index_select(0, r).clamp(0.0, 1.0)
        j = self.J.index_select(0, r)
        b = t.bernoulli(q)
        oq = r.mul(b.long())
        oj = j.mul((1 - b).long())

        return oq + oj
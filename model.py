import torch.nn as nn
import numpy as np


class SkipGramEmbeddings(nn.Module):

    def __init__(self, vocab_size, embed_len):
        super(SkipGramEmbeddings, self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embed_len)#, sparse=True)
        #self.context_embeds = nn.Embedding(vocab_size, embed_len)# sparse=True)

    def forward(self, center, context):
        """
        Acts as a lookup for the center and context words' embeddings

        :param center: The center word index
        :param context: The context word index
        :return: The embedding of the target word
        """
        return self.word_embeds(center), self.word_embeds(context)

    def nearest_neighbors(self, word, dictionary):
        """
        Finds vector closest to word_idx vector
        :param word_idx: Integer
        :return: Integer corresponding to word vector in self.word_embeds
        """
        vectors = self.word_embeds.weight.data.cpu().numpy()
        index = dictionary.token2id[word]
        query = vectors[index]

        ranks = vectors.dot(query).squeeze()
        denom = query.T.dot(query).squeeze()
        denom = denom * np.sum(vectors ** 2, 1)
        denom = np.sqrt(denom)
        ranks = ranks / denom
        mostSimilar = []
        [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
        nearest_neighbors = mostSimilar[:10]
        nearest_neighbors = [dictionary[comp] for comp in nearest_neighbors]

        return nearest_neighbors


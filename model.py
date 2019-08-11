import torch.nn as nn


class SkipGramEmbeddings(nn.Module):

    def __init__(self, vocab_size, embed_len):
        super(SkipGramEmbeddings, self).__init__()
        self.word_embeds = nn.Embedding(vocab_size, embed_len)
        self.context_embeds = nn.Embedding(vocab_size, embed_len)

    def forward(self, center, context):
        """
        Acts as a lookup for the center and context words' embeddings

        :param center: The center word index
        :param context: The context word index
        :return: The embedding of the target word
        """
        return self.word_embeds(center), self.context_embeds(context)

import torch.nn as nn

from models.embedding_layer import EmbeddingLayer
from models.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.emb = EmbeddingLayer(args)
        self.encoder = Encoder(args)

    def forward(self, x):
        x = self.emb(x.long())
        output = self.encoder(x)

        return output

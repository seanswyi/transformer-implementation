import torch
import torch.nn as nn

from models.embedding_layer import EmbeddingLayer
from models.encoder import Encoder
from models.multihead_attention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.emb = EmbeddingLayer(args)
        self.encoder = Encoder(args)

    def forward(self, x):
        thing = self.encoder(x)
        import pdb; pdb.set_trace()
        pass

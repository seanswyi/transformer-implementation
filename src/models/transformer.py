import torch
import torch.nn as nn

from models.embedding_layer import EmbeddingLayer
from models.encoder import Encoder
from models.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        self.num_stacks = self.args.num_stacks
        self.d_model = self.args.d_model
        self.vocab_size = self.args.vocab_size

        self.emb = EmbeddingLayer(self.args)
        self.encoder_stack = nn.ModuleList([Encoder(self.args) for _ in range(self.num_stacks)])
        self.decoder_stack = nn.ModuleList([Decoder(self.args) for _ in range(self.num_stacks)])

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, tgt):
        encoder_output = self.encode(src)
        decoder_output = self.decode(tgt, encoder_output)
        output = torch.matmul(decoder_output, self.emb.embedding_layer.weight.transpose(1, 0))

        return output

    def encode(self, src):
        src_emb = self.emb(src.long())
        output = self.dropout(src_emb)
        for encoder in self.encoder_stack:
            output = encoder(output)

        return output

    def decode(self, tgt, enc_output):
        tgt_emb = self.emb(tgt.long())
        output = self.dropout(tgt_emb)
        for decoder in self.decoder_stack:
            output = decoder(tgt_emb, enc_output)

        return output

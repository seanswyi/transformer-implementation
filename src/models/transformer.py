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

    def forward(self, src, tgt):
        src_emb = self.emb(src.long())
        tgt_emb = self.emb(tgt.long())
        encoder_output = self.encode(src_emb)
        decoder_output = self.decode(tgt_emb, encoder_output)
        output = torch.matmul(decoder_output, self.emb.embedding_layer.weight.transpose(1, 0))

        return output

    def encode(self, src_emb):
        output = src_emb
        for encoder in self.encoder_stack:
            output = encoder(output)

        return output

    def decode(self, tgt_emb, enc_output):
        output = tgt_emb
        for decoder in self.decoder_stack:
            output = decoder(tgt_emb, enc_output)

        return output

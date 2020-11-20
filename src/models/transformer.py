import torch.nn as nn

from models.embedding_layer import EmbeddingLayer
from models.encoder import Encoder
from models.decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()

        self.emb = EmbeddingLayer(args)
        self.encoder_stack = nn.ModuleList([Encoder(args) for _ in args.num_stacks])
        self.decoder_stack = nn.ModuleList([Decoder(args) for _ in args.num_stacks])
        self.output_linear = nn.Linear(in_features=args.d_model, out_features=args.vocab_size)

    def forward(self, src, tgt):
        src_emb = self.emb(src.long())
        tgt_emb = self.emb(tgt.long())
        encoder_output = self.encoder(src_emb)
        decoder_output = self.decoder(tgt_emb, encoder_output)
        output = self.output_linear(decoder_output)

        return output

    def encode(self, src):
        src_emb = self.emb(src.long())

        output = src_emb
        for encoder in self.encoder_stack:
            output = encoder(output)

        return output

    def decoder(self, tgt, enc_output):
        tgt_emb = self.emb(tgt.long())

        output = tgt_emb
        for decoder in self.decoder_stack:
            output = decoder(tgt_emb, enc_output)

        return output

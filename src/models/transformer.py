from torch import nn

from models.decoder import Decoder
from models.embedding_layer import EmbeddingLayer
from models.encoder import Encoder


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.num_stacks = self.args.num_stacks
        self.d_model = self.args.d_model
        self.vocab_size = self.args.vocab_size

        self.emb = EmbeddingLayer(self.args)

        encoders = [Encoder(self.args) for _ in range(self.num_stacks)]
        self.encoder_stack = nn.Sequential(*encoders)

        decoders = [Decoder(self.args) for _ in range(self.num_stacks)]
        self.decoder_stack = nn.ModuleList(decoders)

        self.output_linear = nn.Linear(
            in_features=self.d_model, out_features=self.vocab_size, bias=False
        )
        self.output_linear.weight = self.emb.embedding_layer.weight

        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, src, tgt):
        src_emb = self.emb(src.long())
        tgt_emb = self.emb(tgt.long())

        enc_output = self.encode(src_emb)
        dec_output = self.decode(tgt_emb, enc_output)

        output2 = self.dropout(self.output_linear(dec_output))

        return self.softmax(output2)

    def encode(self, src):
        output = self.encoder_stack(src)
        return output

    def decode(self, tgt, enc_output):
        output = tgt
        for i in range(self.num_stacks):
            output = self.decoder_stack[i](output, enc_output)

        return output

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
        self.output_linear = nn.Linear(in_features=self.d_model, out_features=self.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.dropout = nn.Dropout(p=0.1)

        self.encoder = Encoder(self.args)
        self.two_encoder = nn.Sequential(Encoder(self.args), Encoder(self.args))

    def forward(self, src, tgt):
        src_emb = self.emb(src.long())
        output1 = self.two_encoder(src_emb)
        output2 = self.dropout(self.output_linear(output1))
        return self.softmax(output2)

        # import pdb; pdb.set_trace()
        # encoder_output = self.encode(src)
        # decoder_output = self.decode(tgt, encoder_output)
        # # output = self.output_linear(self.dropout(decoder_output))
        # output = torch.matmul(decoder_output, self.emb.embedding_layer.weight.transpose(1, 0))
        # output = self.softmax(output)

        # return output

    def encode(self, src):
        src_emb = self.emb(src.long())
        output = src_emb
        for i in range(self.num_stacks):
            output = self.encoder_stack[i](output)

        return output

    def decode(self, tgt, enc_output):
        tgt_emb = self.emb(tgt.long())
        output = tgt_emb
        for i in range(self.num_stacks):
            output = self.decoder_stack[i](output, enc_output)

        return output

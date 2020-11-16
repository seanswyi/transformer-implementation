import torch
import torch.nn as nn

from models.multihead_attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.ffnn = nn.Linear(in_features=args.d_model, out_features=args.d_model)
        self.layernorm = nn.LayerNorm(normalized_shape=args.d_model)
        self.multiheadattention = MultiHeadAttention(args)

    def forward(self, x):
        attn_output = self.multiheadattention(x, x, x)
        output1 = self.layernorm(x + attn_output)
        output2 = self.layernorm(output1 + self.ffnn)

        return output2

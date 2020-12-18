import torch.nn as nn

from models.feedforwardnn import FeedForwardNN
from models.multihead_attention import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.ffnn = FeedForwardNN(args)
        self.layernorm = nn.LayerNorm(normalized_shape=args.d_model)
        self.multihead_attention = MultiHeadAttention(args)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        attn_output = self.dropout(self.multihead_attention(x, x, x))
        output1 = self.layernorm(x + attn_output)
        output2 = self.layernorm(output1 + self.dropout(self.ffnn(output1)))

        return output2

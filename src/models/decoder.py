import torch.nn as nn

from models.feedforwardnn import FeedForwardNN
from models.multihead_attention import MultiHeadAttention


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.ffnn = FeedForwardNN(args)
        self.layernorm = nn.LayerNorm(normalized_shape=args.d_model)
        self.multihead_attention = MultiHeadAttention(args)
        self.masked_multihead_attention = MultiHeadAttention(args, mask=True)

    def forward(self, x, enc_x):
        attn_output = self.masked_multihead_attention(x, x, x)
        output1 = self.layernorm(x + attn_output)

        attn_output2 = self.multihead_attention(x, enc_x, enc_x)
        output2 = self.layernorm(attn_output2 + output1)

        output3 = self.layernorm(output2 + self.ffnn(output2))

        return output3

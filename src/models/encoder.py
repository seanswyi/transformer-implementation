import torch
from torch import nn

from models.feedforwardnn import FeedForwardNN
from models.multihead_attention import MultiHeadAttention


class Encoder(nn.Module):
    """
    Encoder object/sublayer.

    Attributes (in alphabetical order)
    ----------------------------------
    dropout: <torch.nn.modules.dropout.Dropout> Dropout from https://jmlr.org/papers/v15/srivastava14a.html.
    ffnn: <models.feedforwardnn.FeedForwardNN> Two-layer feedforward neural network with ReLU activation in between layers.
    layernorm: <torch.nn.modules.normalization.LayerNorm> Layer normalization from https://arxiv.org/abs/1607.06450.
    multihead_attention: <models.multihead_attention.MultiHeadAttention> Multihead Attention to perform self-attention.
    """

    def __init__(self, args):
        """
        Basic initialization of Encoder.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        super().__init__()

        self.ffnn = FeedForwardNN(args)
        self.layernorm = nn.LayerNorm(normalized_shape=args.d_model)
        self.multihead_attention = MultiHeadAttention(args=args)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for encoding.

        Arguments
        ---------
        x: <torch.Tensor> Target data.

        Returns
        -------
        <torch.Tensor> Output after one layer of decoding.
        """
        attn_output = self.dropout(self.multihead_attention(x, x, x, mask=mask))
        output1 = self.layernorm(x + attn_output)
        output2 = self.layernorm(output1 + self.dropout(self.ffnn(output1)))

        return output2

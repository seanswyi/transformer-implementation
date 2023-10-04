import torch
from torch import nn

from models.feedforwardnn import FeedForwardNN
from models.multihead_attention import MultiHeadAttention


class Decoder(nn.Module):
    """
    Decoder object/sublayer.

    Attributes (in alphabetical order)
    ----------------------------------
    dropout: <torch.nn.modules.dropout.Dropout> Dropout from https://jmlr.org/papers/v15/srivastava14a.html.
    ffnn: <models.feedforwardnn.FeedForwardNN> Two-layer feedforward neural network with ReLU activation in between layers.
    layernorm: <torch.nn.modules.normalization.LayerNorm> Layer normalization from https://arxiv.org/abs/1607.06450.
    masked_multihead_attention: <models.multihead_attention.MultiHeadAttention> Multihead Attention with mask after QK operation.
    multihead_attention: <models.multihead_attention.MultiHeadAttention> Multihead Attention to perform self-attention.
    """

    def __init__(self, args):
        """
        Basic initialization of Decoder.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.
        """
        super().__init__()

        self.ffnn = FeedForwardNN(args)
        self.layernorm = nn.LayerNorm(normalized_shape=args.d_model)
        self.multihead_attention = MultiHeadAttention(args=args)
        self.masked_multihead_attention = MultiHeadAttention(args=args)
        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        x: torch.Tensor,
        enc_x: torch.Tensor,
        combined_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for decoding.

        Arguments
        ---------
        x: <torch.Tensor> Target data.
        enc_x: <torch.Tensor> Output from the Encoder stack.

        Returns
        -------
        <torch.Tensor> Output after one layer of decoding.
        """
        attn_output = self.masked_multihead_attention(x, x, x, mask=combined_mask)
        output1 = self.layernorm(x + attn_output)

        attn_output2 = self.multihead_attention(x, enc_x, enc_x, mask=padding_mask)
        output2 = self.layernorm(attn_output2 + output1)

        output3 = self.layernorm(output2 + self.dropout(self.ffnn(output2)))

        return output3

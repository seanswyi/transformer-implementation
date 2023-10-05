import argparse
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger()


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    d_k: int,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Function to perform scaled dot-product attention.

    Arguments
    ---------
    q: <torch.Tensor> Query matrix.
    k: <torch.Tensor> Key matrix.
    v: <torch.Tensor> Value matrix.
    d_k: <int> Dimensionality for linear projection of attention. d_model / num_heads = 64 in this implementation.

    Keyword Arguments
    -----------------
    mask: <bool> If True, then set the mask after the matrix multiplication QK.

    Returns
    -------
    output: <torch.Tensor> Output of scaled dot-product self-attention.

    Be careful of how you set your mask. You want to make sure that the masked values are an "extremely negative" value \
        in order for the last softmax to work properly.
    """
    qk = torch.matmul(q, k.transpose(2, 1))
    qk /= np.sqrt(d_k)

    if mask is not None:
        mask = mask.to(qk.device)

        # We have to unsqueeze padding mask for broadcasting.
        try:
            qk += mask
        except RuntimeError:
            qk += mask.unsqueeze(-1)

    qk = F.softmax(qk, dim=-1)
    output = torch.matmul(qk, v)

    return output


class SingleHeadAttention(nn.Module):
    """
    Object to hold a self-attention operation for one head.

    Attributes (in alphabetical order)
    ----------------------------------
    WK: <torch.nn.modules.linear.Linear> Linear projection for keys.
    WQ: <torch.nn.modules.linear.Linear> Linear projection for queries.
    WV: <torch.nn.modules.linear.Linear> Linear projection for values.
    d_k: <int> Dimensionality for linear projection of attention. d_model / num_heads = 64 in this implementation.
    d_model <int> Dimensionality of model. 512 as per the paper.
    d_q: <int> Dimensionality for linear projection of attention. d_model / num_heads = 64 in this implementation.
    d_v: <int> Dimensionality for linear projection of attention. d_model / num_heads = 64 in this implementation.
    dropout: <torch.nn.modules.dropout.Dropout> Dropout from https://jmlr.org/papers/v15/srivastava14a.html.
    mask: <bool> If True, then set the mask after the matrix multiplication QK.
    num_heads: <int> Number of heads. 8 as per the paper.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Basic initialization of SingleHeadAttention.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.

        Keyword Arguments
        -----------------
        mask: <bool> If True, then set the mask after the matrix multiplication QK.
        """
        super().__init__()

        self.d_model = args.d_model
        self.num_heads = args.num_heads

        self.d_q = int(self.d_model / self.num_heads)
        self.d_k = int(self.d_model / self.num_heads)
        self.d_v = int(self.d_model / self.num_heads)

        self.WQ = nn.Linear(in_features=self.d_model, out_features=self.d_q)
        self.WK = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.WV = nn.Linear(in_features=self.d_model, out_features=self.d_v)

        self.dropout = nn.Dropout(p=0.1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for SingleHeadAttention.

        Arguments
        ---------
        q: <torch.Tensor> Query matrix.
        k: <torch.Tensor> Key matrix.
        v: <torch.Tensor> Value matrix.

        Returns
        -------
        output: <torch.Tensor> Output after performing self-attention.
        """
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        output = self.dropout(attention(q, k, v, d_k=self.d_k, mask=mask))

        return output


class MultiHeadAttention(nn.Module):
    """
    Object to hold a self-attention operation for multiple heads.

    Attributes (in alphabetical order)
    ----------------------------------
    WO: <torch.nn.modules.linear.Linear> Linear projection for output.
    attention_heads: <torch.nn.modules.container.ModuleList> Object to hold head's self-attentionv. 8 here as per the paper.
    d_model: <int> Dimensionality of model. 512 as per the paper.
    d_v: <int> Dimensionality for linear projection of attention. d_model / num_heads = 64 in this implementation.
    mask: <bool> If True, then set the mask after the matrix multiplication QK.
    num_heads: <int> Number of heads. 8 as per the paper.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """
        Basic initialization of MultiHeadAttention.

        Arguments
        ---------
        args: <argparse.Namespace> Arguments used for overall process.

        Keyword Arguments
        -----------------
        mask: <bool> If True, then set the mask after the matrix multiplication QK.
        """
        super().__init__()

        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.d_v = int(self.d_model / self.num_heads)

        self.WO = nn.Linear(
            in_features=(self.num_heads * self.d_v),
            out_features=self.d_model,
        )
        nn.init.xavier_uniform_(self.WO.weight)

        self.attention_heads = nn.ModuleList(
            [SingleHeadAttention(args) for _ in range(self.num_heads)]
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward pass for MultiHeadAttention.

        Arguments
        ---------
        q: <torch.Tensor> Query matrix.
        k: <torch.Tensor> Key matrix.
        v: <torch.Tensor> Value matrix.

        Returns
        -------
        output: <torch.Tensor> Output after performing self-attention for all head and linear projecting the output.
        """
        attention_results = [head(q, k, v, mask=mask) for head in self.attention_heads]
        attention_concatenated = torch.cat(attention_results, dim=2)
        output = self.WO(attention_concatenated)

        return output

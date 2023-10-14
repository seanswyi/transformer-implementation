import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


logger = logging.getLogger()


def attention(q, k, v, d_k, mask=False):
    qk = torch.matmul(q, k.transpose(2, 1))
    qk /= np.sqrt(d_k)

    if mask:
        ones = torch.ones(size=qk.shape)
        mask_matrix = torch.triu(ones, diagonal=1) * (-1e9)

        if qk.is_cuda:
            mask_matrix = mask_matrix.to("cuda")
        else:
            logger.warning("Not using GPU!")

        qk += mask_matrix

    qk = F.softmax(qk, dim=-1)

    output = torch.matmul(qk, v)

    return output


class SingleHeadAttention(nn.Module):
    def __init__(self, args, mask=False):
        super().__init__()

        self.mask = mask
        self.d_model = args.d_model
        self.num_heads = args.num_heads

        self.d_q = int(self.d_model / self.num_heads)
        self.d_k = int(self.d_model / self.num_heads)
        self.d_v = int(self.d_model / self.num_heads)

        self.WQ = nn.Linear(in_features=self.d_model, out_features=self.d_q)
        self.WK = nn.Linear(in_features=self.d_model, out_features=self.d_k)
        self.WV = nn.Linear(in_features=self.d_model, out_features=self.d_v)

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v):
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        output = self.dropout(attention(q, k, v, d_k=self.d_k, mask=self.mask))

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, args, mask=False):
        super().__init__()

        self.mask = mask
        self.d_model = args.d_model
        self.num_heads = args.num_heads
        self.d_v = int(self.d_model / self.num_heads)

        self.WO = nn.Linear(
            in_features=(self.num_heads * self.d_v), out_features=self.d_model
        )
        nn.init.xavier_uniform_(self.WO.weight)

        self.attention_heads = nn.ModuleList(
            [SingleHeadAttention(args, mask=self.mask) for _ in range(self.num_heads)]
        )

    def forward(self, q, k, v):
        attention_results = [head(q, k, v) for head in self.attention_heads]
        attention_concatenated = torch.cat(attention_results, dim=2)
        output = self.WO(attention_concatenated)

        return output

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v, d_k=512, mask=False):
    qk = torch.matmul(q, k.T)
    qk /= np.sqrt(d_k)

    if mask:
        ones = torch.ones(size=qk.shape)
        mask_matrix = torch.tril(ones)
        qk *= mask_matrix

    qk = F.softmax(qk, dim=0)
    output = torch.matmul(qk, v)

    return output


class SingleHeadAttention(nn.Module):
    def __init__(self, args, mask=False):
        super(SingleHeadAttention, self).__init__()

        self.mask = mask

        self.WQ = nn.Linear(in_features=args.d_model, out_features=args.d_k)
        self.WK = nn.Linear(in_features=args.d_model, out_features=args.d_k)
        self.WV = nn.Linear(in_features=args.d_model, out_features=args.d_v)

    def forward(self, q, k, v):
        q = self.WQ(q)
        k = self.WK(k)
        v = self.WV(v)

        output = attention(q, k, v, mask=self.mask)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, args, mask=False):
        super(MultiHeadAttention, self).__init__()

        self.mask = mask
        self.num_heads = args.num_heads
        self.WO = nn.Linear(in_features=(self.num_heads * args.d_v), out_features=args.d_model)

        self.single_head_attention = SingleHeadAttention(args, mask=self.mask)

    def forward(self, q, k, v):
        attention_results = []

        for _ in range(self.num_heads):
            attention_output = self.single_head_attention(q, k, v)
            attention_results.append(attention_output.clone().detach())

        attention_concatenated = torch.cat(attention_results, dim=1)
        output = self.WO(attention_concatenated)

        return output

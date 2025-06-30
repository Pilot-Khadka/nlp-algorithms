import math
import torch
import torch.nn as nn


def scaled_dot_product(query, key, value, mask=None):
    # b, l, d * b, d, l -> b, l, l
    # attention between seq. len
    # i.e between every word
    score = torch.matmul(query, key.transpose(-1, -2))
    score = score / math.sqrt(query.size(-1))  # dim

    if mask is not None:
        score = score.masked_fill(mask == 0, float("-inf"))

    weights = torch.softmax(score, dim=-1)
    return weights @ value


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Parameter(torch.randn(d_model, d_model))
        self.w_k = nn.Parameter(torch.randn(d_model, d_model))
        self.w_v = nn.Parameter(torch.randn(d_model, d_model))
        self.w_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, q, k, v, mask=None):
        batch_size, seq_length, dim = q.size()

        def transform(x, w):
            # x -> b, l, d
            # w -> d, d
            # out - > b, d, d
            x = x @ w
            # num_head * d_k = d_model
            x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
            # out -> [batch_size, num_heads, seq_length, d_k]
            return x.transpose(1, 2)

        q = transform(q, self.w_q)
        k = transform(k, self.w_k)
        v = transform(v, self.w_v)

        out = scaled_dot_product(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)
        return out @ self.w_o

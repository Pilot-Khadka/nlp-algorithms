import torch
import torch.nn as nn


def scaled_dot_product(query, key, value, mask=None):
    scale = query.size(-1) ** -0.5
    score = (query @ key.transpose(-2, -1)) * scale

    if mask is not None:
        score = score + mask

    return (score.softmax(dim=-1)) @ value


class MultiHeadAttentionNaive(nn.Module):
    """
    Textbook implementation of multiheaded attention,
    slower because matrix multiplication is separated
    """

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

        def transform(x, w, batch_size, seq_length):
            # x -> b, l, d
            # w -> d, d
            # out - > b, d, d
            x = x @ w
            # num_head * d_k = d_model
            x = x.view(batch_size, seq_length, self.num_heads, self.d_k)
            # out -> [batch_size, num_heads, seq_length, d_k]
            return x.transpose(1, 2)

        q = transform(q, self.w_q, batch_size, seq_length)
        k = transform(k, self.w_k, batch_size, seq_length)
        v = transform(v, self.w_v, batch_size, seq_length)

        out = scaled_dot_product(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)
        return out @ self.w_o

import math
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        pass


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

        # combined weight matrix for Q, K, and V
        # single matrix of size (d_model, 3 * d_model)
        self.w_qkv = nn.Parameter(torch.randn(d_model, 3 * d_model))
        self.w_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, x, mask=None):
        batch_size, seq_length, dim = x.size()

        # output tensor will have dim: (batch_size, seq_length, 3 * d_model)
        qkv = x @ self.w_qkv

        q, k, v = torch.chunk(qkv, 3, dim=-1)

        def reshape_and_transpose(tensor):
            return tensor.view(
                batch_size, seq_length, self.num_heads, self.d_k
            ).transpose(1, 2)

        q = reshape_and_transpose(q)
        k = reshape_and_transpose(k)
        v = reshape_and_transpose(v)

        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scaled_dot_product.masked_fill_(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scaled_dot_product, dim=-1)

        out = torch.matmul(attention_weights, v)

        # concat heads and apply final linear layer
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_length, dim)
        return out @ self.w_o


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


if __name__ == "__main__":
    model_dim = 512

    batch_size = 8
    sequence_length = 10  # number of tokens

    # [b, l, D]
    query = torch.randn(batch_size, sequence_length, model_dim)
    key = torch.randn(batch_size, sequence_length, model_dim)
    value = torch.randn(batch_size, sequence_length, model_dim)
    attention1 = scaled_dot_product(query, key, value)

    attention = MultiHeadAttention(d_model=model_dim, num_heads=8)
    attention.forward(query, key, value)

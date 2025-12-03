import math
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    """
    additive attention doesn’t directly compare query/key vectors like dot-product attention
    it projects them into some shared space, squashes them with a nonlinearity
    then reduces to a scalar score
    """

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_ff, bias=False)
        self.W_k = nn.Linear(d_model, d_ff, bias=False)
        self.v = nn.Linear(d_ff, 1, bias=False)

    def forward(self, Q, K, V, mask=None):
        # Q: [batch_size, decoder_len, model_dim]
        # K, V: [batch_size, encoder_len, model_dim]

        # for pair-wise combination between every decoder position and encoder position
        # eg:
        # Q.unsqueeze(2) -> [1, 2, 1, 3]
        # K.unsqueeze(1) ->  [1, 1, 4, 3]
        # result: [1, 2, 4, 3] through broadcasting

        q_proj = self.W_q(Q).unsqueeze(2)
        k_proj = self.W_k(K).unsqueeze(1)
        energy = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)

        if mask is not None:
            energy.masked_fill_(mask.unsqueeze(1) == 0, float("-inf"))

        attn_weights = torch.softmax(energy, dim=-1)
        context = torch.matmul(attn_weights, V)  # [batch, dec_len, d_model]

        return context, attn_weights


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q comes from the decoder so it has its own weight matrix
        self.w_q = nn.Parameter(torch.randn(d_model, d_model))
        # K and V come from the encoder
        self.w_kv = nn.Parameter(torch.randn(d_model, 2 * d_model))
        self.w_o = nn.Parameter(torch.randn(d_model, d_model))

    def forward(self, decoder_out, encoder_out, mask=None):
        batch_size, dec_seq_length, dim = decoder_out.size()
        enc_seq_length = encoder_out.size(1)

        q = decoder_out @ self.w_q
        kv = encoder_out @ self.w_kv
        k, v = torch.chunk(kv, 2, dim=-1)

        def reshape_and_transpose(tensor, seq_length):
            return tensor.view(
                batch_size, seq_length, self.num_heads, self.d_k
            ).transpose(1, 2)

        q = reshape_and_transpose(q, dec_seq_length)
        k = reshape_and_transpose(k, enc_seq_length)
        v = reshape_and_transpose(v, enc_seq_length)

        scaled_dot_product = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            # mask should have shape (batch_size, 1, 1, enc_seq_length)
            scaled_dot_product.masked_fill_(mask == 0, float("-inf"))

        attention_weights = torch.softmax(scaled_dot_product, dim=-1)
        out = torch.matmul(attention_weights, v)

        out = out.transpose(1, 2).contiguous().view(batch_size, dec_seq_length, dim)
        return out @ self.w_o


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
    sequence_length = 10

    x = torch.randn(batch_size, sequence_length, model_dim)

    attention = MultiHeadAttention(d_model=model_dim, num_heads=8)
    out = attention(x)

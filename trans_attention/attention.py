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
        self.scale = self.d_k**-0.5

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_out, encoder_out, mask=None):
        B, dec_len, D = decoder_out.shape
        enc_len = encoder_out.shape[1]

        # Q
        q = self.w_q(decoder_out).reshape(B, dec_len, self.num_heads, self.d_k)
        q = q.transpose(1, 2)  # (B, H, dec_len, d_k)

        # K, V
        kv = self.w_kv(encoder_out).reshape(B, enc_len, 2, self.num_heads, self.d_k)
        kv = kv.permute(2, 0, 3, 1, 4)  # (2, B, H, enc_len, d_k)
        k, v = kv[0], kv[1]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Expect mask shape: (B, 1, dec_len, enc_len)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        # Merge heads
        out = out.transpose(1, 2).reshape(B, dec_len, D)
        return self.w_o(out)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = self.d_k**-0.5

        self.w_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, S, D = x.shape

        qkv = self.w_qkv(x).reshape(B, S, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Expect mask shape: (B, 1, S, S)
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).reshape(B, S, D)
        return self.w_o(out)


if __name__ == "__main__":
    model_dim = 512
    batch_size = 8
    sequence_length = 10

    x = torch.randn(batch_size, sequence_length, model_dim)

    attention = MultiHeadAttention(d_model=model_dim, num_heads=8)
    out = attention(x)

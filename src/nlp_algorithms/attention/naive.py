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


def scaled_dot_product(query, key, value, mask=None):
    scale = query.size(-1) ** -0.5
    score = (query @ key.transpose(-2, -1)) * scale

    if mask is not None:
        score = score + mask

    return (score.softmax(dim=-1)) @ value


class MultiHeadAttention(nn.Module):
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

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must split evenly"

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def _shape(self, x, B, S):
        # (B, S, D) --> (B, num_heads, S, head_dim)
        return x.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states,  # (B, tgt_len, d_model)
        key_value_states=None,
        attention_mask=None,
    ):
        is_cross = key_value_states is not None

        B, tgt_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)

        if is_cross:
            key = self.k_proj(key_value_states)
            value = self.v_proj(key_value_states)
            src_len = key_value_states.size(1)
        else:
            key = self.k_proj(hidden_states)
            value = self.v_proj(hidden_states)
            src_len = tgt_len

        # (B, heads, seq_len, head_dim)
        query = self._shape(query, B, tgt_len)
        key = self._shape(key, B, src_len)
        value = self._shape(value, B, src_len)

        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        # shape: (B, heads, tgt_len, src_len)

        if attention_mask is not None:
            # masks are additive (0 or -inf)
            # Must be broadcastable to (B, heads, tgt_len, src_len)
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, value)
        # (B, heads, tgt_len, head_dim)

        # merge heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, tgt_len, -1)

        return self.out_proj(attn_output)


def make_padding_mask(seq, pad_idx=0):
    """
    seq: (B,S)
    output: (B,1,1,S)
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def make_causal_mask(T):
    """
    output: (1, 1, T, T)
    """
    return torch.triu(torch.ones(T, T), diagonal=1).bool().unsqueeze(0).unsqueeze(0)


def convert_to_additive(mask_bool):
    """
    boolean mask -> additive mask
    True = keep (0)
    False = mask (-inf)
    """
    return mask_bool.masked_fill(~mask_bool, float("-inf")).float()


def test_self_attention_no_mask():
    print("TEST: Self-attention no mask")
    batch_size, seq_len, dim = 16, 10, 64
    num_heads = 8
    x = torch.randn(batch_size, seq_len, dim)
    mha = MultiHeadAttention(d_model=dim, num_heads=num_heads)

    out = mha(x)
    assert out.shape == (batch_size, seq_len, dim)
    assert not torch.isnan(out).any()
    print(" OK!")


def test_self_attention_padding_mask():
    print("TEST: Self-attention padding mask")
    B, T, D = 3, 12, 32
    x = torch.randn(B, T, D)
    seq = torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0], [7, 8, 9, 10, 11, 12]])
    seq = F.pad(seq, (0, T - seq.size(1)), value=0)

    padding_mask = convert_to_additive(make_padding_mask(seq))

    mha = MultiHeadAttention(D, 4)
    out = mha(x, attention_mask=padding_mask)

    assert out.shape == (B, T, D)
    assert not torch.isnan(out).any()
    print(" OK!")


def test_self_attention_causal_mask():
    print("TEST: Self-attention causal mask")
    B, T, D = 2, 6, 16
    x = torch.randn(B, T, D)

    causal_mask = convert_to_additive(~make_causal_mask(T))

    mha = MultiHeadAttention(D, 4)
    out = mha(x, attention_mask=causal_mask)

    assert out.shape == (B, T, D)
    print(" OK!")


def test_combined_causal_padding_mask():
    print("TEST: Combined causal + padding")
    B, T, D = 2, 8, 32

    x = torch.randn(B, T, D)
    # shape:(2,8)
    seq = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [9, 8, 7, 6, 5, 4, 0, 0]])

    # shape:(2,1,1,8)
    padding = make_padding_mask(seq)
    # shape:(1,1,8,8)
    causal = ~make_causal_mask(T)  # True = keep
    # shape:(2,1,8,8)
    combined = convert_to_additive(padding & causal)

    mha = MultiHeadAttention(D, 4)
    out = mha(x, attention_mask=combined)

    assert out.shape == (B, T, D)
    print(" OK!")


def test_cross_attention_padding_mask():
    print("TEST: Cross-attention padding mask")
    B, S, T, D = 2, 12, 6, 32

    src = torch.randn(B, S, D)
    tgt = torch.randn(B, T, D)

    # shape:(2,12)
    src_tokens = torch.tensor(
        [[1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0], [4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0]]
    )

    # shape:(2,1,1,12)
    padding_mask = convert_to_additive(make_padding_mask(src_tokens))  # (B,1,1,S)

    mha = MultiHeadAttention(D, 4)
    out = mha(tgt, key_value_states=src, attention_mask=padding_mask)
    assert out.shape == (B, T, D)
    print(" OK!")


if __name__ == "__main__":
    test_self_attention_no_mask()
    test_self_attention_padding_mask()
    test_self_attention_causal_mask()
    test_combined_causal_padding_mask()
    test_cross_attention_padding_mask()

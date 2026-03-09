import torch.nn as nn

from nlp_algorithms.attention import MultiHeadAttention
from .sublayers import FeedForward, ResidualBlock


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            # pyrefly: ignore [unexpected-keyword]
            dropout=dropout,
        )
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # each block handles: norm -> sublayer -> dropout -> residual
        self.attn_block = ResidualBlock(d_model=d_model, dropout=dropout)
        self.ff_block = ResidualBlock(d_model=d_model, dropout=dropout)

    def forward(self, x, mask=None):
        def attn_sublayer(norm_x):
            return self.self_attn(
                norm_x,
                key_value_states=None,
                attention_mask=mask,
            )

        x = self.attn_block(x, attn_sublayer)
        x = self.ff_block(x, self.ff)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            # pyrefly: ignore [unexpected-keyword]
            dropout=dropout,
        )
        self.cross_attn = MultiHeadAttention(
            # pyrefly: ignore [unexpected-keyword]
            d_model=d_model, num_heads=num_heads, dropout=dropout
        )
        self.ff = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)

        # each block handles: norm -> sublayer -> dropout -> residual
        self.self_attn_block = ResidualBlock(d_model=d_model, dropout=dropout)
        self.cross_attn_block = ResidualBlock(d_model=d_model, dropout=dropout)
        self.ff_block = ResidualBlock(d_model=d_model, dropout=dropout)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # ---- masked self-attention ----
        def self_attn_sublayer(norm_x):
            return self.self_attn(
                norm_x,
                key_value_states=None,
                attention_mask=tgt_mask,
            )

        x = self.self_attn_block(x, self_attn_sublayer)

        # ---- encoder-decoder cross-attention ----
        def cross_attn_sublayer(norm_x):
            return self.cross_attn(
                norm_x,
                key_value_states=encoder_out,
                attention_mask=src_mask,
            )

        x = self.cross_attn_block(x, cross_attn_sublayer)

        # ---- feed-forward ----
        x = self.ff_block(x, self.ff)

        return x

import torch
import torch.nn as nn

from common.layernorm import LayerNorm
from common.positional_encoding import PositionalEncoding
from a09_attention.attention import MultiHeadAttention, MultiHeadCrossAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)

        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_out, source_mask=None, target_mask=None):
        # self-attention block with residual connection
        attention_out = self.attention(self.norm1(x), target_mask)
        x = x + attention_out

        # corss-attention block with residual connection
        cross_attention_out = self.cross_attention(
            self.norm2(x), self.norm2(encoder_out), source_mask
        )
        x = x + cross_attention_out

        # feed-forward block with residual connection
        ff_out = self.ff(self.norm3(x))
        x = x + ff_out
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, target, encoder_out, source_mask=None, target_mask=None):
        x = self.pos(self.embedding(target))
        for layer in self.layers:
            x = layer(x, encoder_out, source_mask, target_mask)
        return self.out(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        normalized_x = self.norm1(x)
        attention_out = self.attention(normalized_x, mask)
        x = x + attention_out

        normalized_x = self.norm2(x)
        ff_out = self.ff(normalized_x)
        x = x + ff_out
        return x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, source, source_mask=None):
        x = self.pos(self.embedding(source))
        for layer in self.layers:
            x = layer(x, source_mask)

        return x


class Seq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab,
        target_vocab,
        d_model=512,
        num_layers=8,
        num_heads=8,
        d_ff=2048,
    ):
        super().__init__()
        self.encoder = EncoderBlock(source_vocab, d_model, num_layers, num_heads, d_ff)
        self.decoder = DecoderBlock(target_vocab, d_model, num_layers, num_heads, d_ff)

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoder_out = self.encoder(source, source_mask)
        return self.decoder(target, encoder_out, source_mask, target_mask)


if __name__ == "__main__":
    src = torch.randint(0, 100, (32, 20))  # [batch, src_len]
    tgt = torch.randint(0, 100, (32, 20))  # [batch, tgt_len]

    model = Seq2Seq(source_vocab=100, target_vocab=100)
    out = model(src, tgt)
    print(out.shape)  # [batch, tgt_len, vocab]

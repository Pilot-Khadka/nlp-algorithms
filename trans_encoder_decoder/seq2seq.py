import torch
import torch.nn as nn

from engine.registry import register_model
from net_common.positional_encoding import PositionalEncoding
from trans_attention.attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)

        self.attn_block = ResidualBlock(d_model)
        self.ff_block = ResidualBlock(d_model)

    def forward(self, x, mask=None):
        x = self.attn_block(
            x,
            lambda norm_x: self.self_attn(
                norm_x,
                key_value_states=None,
                attention_mask=mask,
            ),
        )
        x = self.ff_block(x, self.ff)
        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def forward(self, src, src_mask=None):
        x = self.position(self.embedding(src))

        for layer in self.layers:
            x = layer(x, src_mask)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(
            d_model, num_heads
        )  # cross attn handled internally
        self.ff = FeedForward(d_model, d_ff)

        self.self_attn_block = ResidualBlock(d_model)
        self.cross_attn_block = ResidualBlock(d_model)
        self.ff_block = ResidualBlock(d_model)

    def forward(self, x, encoder_out, src_mask=None, tgt_mask=None):
        # maksed self-attention
        x = self.self_attn_block(
            x,
            lambda norm_x: self.self_attn(
                norm_x,
                key_value_states=None,
                attention_mask=tgt_mask,
            ),
        )

        # cross-attention
        x = self.cross_attn_block(
            x,
            lambda norm_x: self.cross_attn(
                norm_x,
                key_value_states=encoder_out,
                attention_mask=src_mask,
            ),
        )

        x = self.ff_block(x, self.ff)
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.position = PositionalEncoding(d_model)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, encoder_out, src_mask=None, tgt_mask=None):
        x = self.position(self.embedding(tgt))

        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)

        return self.output_projection(x)


@register_model("seq2seq")
class Seq2Seq(nn.Module):
    def __init__(
        self,
        source_vocab_size,
        target_vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
    ):
        super().__init__()

        self.encoder = Encoder(source_vocab_size, d_model, num_layers, num_heads, d_ff)
        self.decoder = Decoder(target_vocab_size, d_model, num_layers, num_heads, d_ff)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        encoder_out = self.encoder(src, src_mask)
        return self.decoder(tgt, encoder_out, src_mask, tgt_mask)


def test_seq2seq_shapes():
    batch_size = 32
    src_len = 20
    tgt_len = 15
    vocab_size_src = 100
    vocab_size_tgt = 120
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128

    src = torch.randint(0, vocab_size_src, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size_tgt, (batch_size, tgt_len))

    model = Seq2Seq(
        source_vocab_size=vocab_size_src,
        target_vocab_size=vocab_size_tgt,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )

    out = model(src, tgt)

    assert out.shape == (batch_size, tgt_len, vocab_size_tgt), (
        f"Expected {(batch_size, tgt_len, vocab_size_tgt)}, got {out.shape}"
    )

    print("Forward pass shape test passed!")


def test_seq2seq_masking():
    batch_size = 2
    src_len = 5
    tgt_len = 5
    vocab_size = 10

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    tgt_mask = (
        torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0).repeat(batch_size, 1, 1)
    )

    model = Seq2Seq(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        d_model=32,
        num_layers=1,
        num_heads=2,
        d_ff=64,
    )
    out = model(src, tgt, tgt_mask=tgt_mask)

    assert torch.isfinite(out).all(), "Output contains non-finite values"
    print("Masking test passed!")


def test_seq2seq_different_lengths():
    batch_size = 4
    src_len = 7
    tgt_len = 3
    vocab_size = 20

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))

    model = Seq2Seq(
        source_vocab_size=vocab_size,
        target_vocab_size=vocab_size,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
    )
    out = model(src, tgt)

    assert out.shape == (batch_size, tgt_len, vocab_size)
    print("Different length test passed!")


if __name__ == "__main__":
    test_seq2seq_shapes()
    test_seq2seq_masking()
    test_seq2seq_different_lengths()
    print("All tests passed!")

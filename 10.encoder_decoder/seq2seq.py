import os
import torch
import torch.nn as nn
from layernorm import LayerNorm
from positional_encoding import PositionalEncoding
from utils.utils import import_function_from_folder

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)  # ~/nlp-algorithms
ATTN_DIR = os.path.join(BASE_DIR, "09.attention")

MultiHeadAttention = import_function_from_folder(
    folder_path=ATTN_DIR,
    module_filename="attention.py",
    function_name="MultiHeadAttention",
)


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
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, encoder_out, source_mask=None, target_mask=None):
        # query, key, value
        # residual connections
        x = x + self.attention(self.norm1(x), self.norm1(x),
                               self.norm1(x), target_mask)

        # cross-attention uses key from encoder
        x = x + self.cross_attention(
            self.norm2(x), self.norm2(encoder_out), self.norm2(
                encoder_out), source_mask
        )

        x = x + self.ff(self.norm3(x))
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
        # attention - [query, key, value]
        # residual connections
        x = x + self.attention(self.norm1(x),
                               self.norm2(x), self.norm1(x), mask)
        x = x + self.ff(self.norm2(x))
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
        self.encoder = EncoderBlock(
            source_vocab, d_model, num_layers, num_heads, d_ff)
        self.decoder = DecoderBlock(
            target_vocab, d_model, num_layers, num_heads, d_ff)

    def forward(self, source, target, source_mask=None, target_mask=None):
        encoder_out = self.encoder(source, source_mask)
        return self.decoder(target, encoder_out, source_mask, target_mask)


if __name__ == "__main__":
    src = torch.randint(0, 100, (32, 20))  # [batch, src_len]
    tgt = torch.randint(0, 100, (32, 20))  # [batch, tgt_len]

    model = Seq2Seq(source_vocab=100, target_vocab=100)
    out = model(src, tgt)
    print(out.shape)  # [batch, tgt_len, vocab]

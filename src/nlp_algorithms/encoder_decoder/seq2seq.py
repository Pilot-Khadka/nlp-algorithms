import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from .positional_encoding import RotaryPositionalEncoding


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab, dropout: float = 0.0):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(self.lut(x) * math.sqrt(self.d_model))


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ffn: int):
        super().__init__()
        d_ffn = int(2 / 3 * d_ffn)
        self.gate_up = nn.Linear(d_model, 2 * d_ffn, bias=False)
        self.down = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * up)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.ffn = SwiGLUFFN(d_model, d_ffn)
        self.dropout = nn.Dropout(p=dropout)
        self.attn_dropout = dropout

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            mask = mask.unsqueeze(1)

        B, S, _ = x.shape

        residual = x
        x = self.norm1(x)
        q, k, v = (
            self.qkv(x)
            .view(B, S, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = residual + self.out_proj(x.transpose(1, 2).reshape(B, S, -1))

        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)
        self.self_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.self_out = nn.Linear(d_model, d_model, bias=False)
        self.cross_q = nn.Linear(d_model, d_model, bias=False)
        self.cross_kv = nn.Linear(d_model, 2 * d_model, bias=False)
        self.cross_out = nn.Linear(d_model, d_model, bias=False)
        self.ffn = SwiGLUFFN(d_model, d_ffn)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cross_mask is not None:
            cross_mask = cross_mask.unsqueeze(1)

        B, tgt_len, _ = x.shape
        src_len = context.size(1)

        residual = x
        x = self.norm1(x)
        q, k, v = (
            self.self_qkv(x)
            .view(B, tgt_len, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = residual + self.dropout(
            self.self_out(x.transpose(1, 2).reshape(B, tgt_len, -1))
        )

        residual = x
        x = self.norm2(x)
        q = (
            self.cross_q(x)
            .view(B, tgt_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k, v = (
            self.cross_kv(context)
            .view(B, src_len, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        x = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=cross_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = residual + self.dropout(
            self.cross_out(x.transpose(1, 2).reshape(B, tgt_len, -1))
        )

        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ffn=d_ffn,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ffn: int,
        num_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ffn=d_ffn,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        cross_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, context, cross_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        src_embed,
        tgt_embed,
        lm_head,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.lm_head = lm_head

    def encode(self, src, src_mask=None):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, context, tgt, cross_mask=None):
        return self.decoder(self.tgt_embed(tgt), context, cross_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.decode(self.encode(src, src_mask), tgt, src_mask)


def init_weights(model: EncoderDecoder, num_layers: int) -> None:
    """
    based on gpt-2, each block adds to residual stream
    without scaling, variance grows with depth
    scale down the weight by 1/sqrt(2N)
    """
    residual_proj_names = {"out_proj", "self_out", "cross_out", "down"}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            std = (
                0.02 / math.sqrt(2 * num_layers)
                if name.split(".")[-1] in residual_proj_names
                else 0.02
            )
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


def make_model(
    src_vocab,
    tgt_vocab,
    N=6,
    d_model=512,
    d_ff=2048,
    h=8,
    dropout=0.1,
):
    c = copy.deepcopy
    # position = PositionalEncoding(d_model=d_model, dropout=dropout)
    position = RotaryPositionalEncoding(d_model=d_model)
    model = EncoderDecoder(
        encoder=Encoder(
            d_model=d_model,
            num_heads=h,
            d_ffn=d_ff,
            num_layers=N,
            dropout=dropout,
        ),
        decoder=Decoder(
            d_model=d_model,
            num_heads=h,
            d_ffn=d_ff,
            num_layers=N,
            dropout=dropout,
        ),
        src_embed=nn.Sequential(
            Embeddings(
                d_model=d_model,
                vocab=src_vocab,
                dropout=dropout,
            ),
            c(position),
        ),
        tgt_embed=nn.Sequential(
            Embeddings(
                d_model=d_model,
                vocab=tgt_vocab,
                dropout=dropout,
            ),
            c(position),
        ),
        lm_head=Generator(d_model, tgt_vocab),
    )
    init_weights(model=model, num_layers=N)
    return model

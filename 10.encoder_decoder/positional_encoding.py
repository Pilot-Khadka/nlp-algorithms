import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Positional encoding as described in "attention is all you need"

    Formula:
    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Where:
    - pos: position of the token in the sequence (0, 1, 2, ...)
    - i: dimension index (0, 1, 2, ..., d_model//2-1)
    - d_model: embedding dimension

    eg: "Hello world" with d_model=512:

    a). word: Hello [1, 512]
      - PE(0, 0) = sin(0 / 10000^(0/512)) = sin(0) = 0
      - PE(0, 1) = cos(0 / 10000^(0/512)) = cos(0) = 1
      - PE(0, 2) = sin(0 / 10000^(2/512)) = sin(0) = 0
      - PE(0, 3) = cos(0 / 10000^(2/512)) = cos(0) = 1
      - ... and so on for all 512 dimensions

    - token "world" at position 1:
      - PE(1, 0) = sin(1 / 10000^(0/512)) = sin(1) ≈ 0.841
      - PE(1, 1) = cos(1 / 10000^(0/512)) = cos(1) ≈ 0.540
      - ... and so on
    """

    def __init__(self, dim, max_len=10000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)

        # exp + log : numerical stability
        # for large dim, 10,000 ^(10023/1024)
        # a^b = exp(b * ln(a))
        # arange(0, dim, 2) -> even terms only
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        # even indices: 0, 2, 4 ..
        pe[:, 0::2] = torch.sin(position * div_term)

        # odd indices: 1, 3, 5 ..
        pe[:, 1::2] = torch.cos(position * div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


if __name__ == "__main__":
    d_model = 512
    seq_len = 10
    batch_size = 2

    pos_enc = PositionalEncoding(d_model, max_len=1000)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

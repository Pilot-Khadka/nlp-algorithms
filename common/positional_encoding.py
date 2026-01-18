import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, method="sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.method = method.lower()
        self.max_len = max_len

        if self.method == "sinusoidal":
            # precompute once
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
                1
            )  # [max_len, 1]
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float)
                * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            # [1, max_len, d_model] for broadcasting
            self.register_buffer("pe", pe.unsqueeze(0))

        elif self.method == "learned":
            # learned positional embeddings
            self.pos_embedding = nn.Embedding(max_len, d_model)
        else:
            raise ValueError("Method must be 'sinusoidal' or 'learned'")

    def forward(self, x):
        seq_len = x.size(1)

        if self.method == "sinusoidal":
            return x + self.pe[:, :seq_len, :]

        elif self.method == "learned":
            positions = torch.arange(seq_len, device=x.device).unsqueeze(
                0
            )  # [1, seq_len]
            return x + self.pos_embedding(positions)


if __name__ == "__main__":
    d_model = 512
    seq_len = 10
    batch_size = 2

    pos_enc_sin = PositionalEncoding(d_model, max_len=1000, method="sinusoidal")
    x = torch.randn(batch_size, seq_len, d_model)
    out_sin = pos_enc_sin(x)
    print(f"Sinusoidal output shape: {out_sin.shape}")

    pos_enc_learned = PositionalEncoding(d_model, max_len=1000, method="learned")
    out_learned = pos_enc_learned(x)
    print(f"Learned output shape: {out_learned.shape}")

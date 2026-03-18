import math
import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for RoPE"

        self.d_model = d_model
        freq = torch.arange(0, d_model, 2, dtype=torch.float32)
        freq = 1.0 / (10000 ** (freq / d_model))
        pos = torch.arange(max_len, dtype=torch.float32)  # (max_len,)

        # outer product
        angles = pos[:, None] * freq[None, :]

        self.sin: torch.Tensor
        self.cos: torch.Tensor
        angles = torch.outer(pos, freq)

        self.register_buffer("sin", torch.sin(angles).to(torch.bfloat16))
        self.register_buffer("cos", torch.cos(angles).to(torch.bfloat16))

    def forward(self, x):
        B, T, D = x.shape

        x_reshaped = x.view(B, T, D // 2, 2)
        x1 = x_reshaped[..., 0]
        x2 = x_reshaped[..., 1]

        sin = self.sin[:T].unsqueeze(0)
        cos = self.cos[:T].unsqueeze(0)

        rot_x1 = x1 * cos - x2 * sin
        rot_x2 = x1 * sin + x2 * cos

        out = torch.stack([rot_x1, rot_x2], dim=-1)
        return out.view(B, T, D)


def time_encoding(fn, warmup=5, runs=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / runs * 1000
    return elapsed


def example_positional():
    pe = PositionalEncoding(20, dropout=0)
    y = pe.forward(torch.zeros(1, 100, 20))

    dims_to_plot = [4, 5, 6, 7]
    positions = np.arange(100)
    y_np = y.detach().numpy()

    plt.figure(figsize=(12, 6))
    for dim in dims_to_plot:
        plt.plot(positions, y_np[0, :, dim], label=f"dim {dim}")
    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.title("Sinusoidal Positional Encoding")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def example_rope():
    rope = RotaryPositionalEncoding(20)
    x = torch.randn(1, 100, 20)
    y = rope(x)

    dims_to_plot = [4, 5, 6, 7]
    positions = np.arange(100)
    y_np = y.detach().numpy()

    plt.figure(figsize=(12, 6))
    for dim in dims_to_plot:
        plt.plot(positions, y_np[0, :, dim], label=f"dim {dim}")
    plt.xlabel("Position")
    plt.ylabel("Rotated Value")
    plt.title("Rotary Positional Encoding (RoPE)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def compare_timing(d_model=64, seq_len=512, batch=32):
    x = torch.randn(batch, seq_len, d_model)
    pe = PositionalEncoding(d_model, dropout=0)
    rope = RotaryPositionalEncoding(d_model)

    t_sin = time_encoding(lambda: pe(x))
    t_rope = time_encoding(lambda: rope(x))

    print(f"{'Encoding':<30} {'Avg time (ms)':>15}")
    print("-" * 47)
    print(f"{'Sinusoidal PE':<30} {t_sin:>15.4f}")
    print(f"{'RoPE':<30} {t_rope:>15.4f}")
    print(f"\nBatch={batch}, seq_len={seq_len}, d_model={d_model}")


if __name__ == "__main__":
    compare_timing()
    example_positional()
    example_rope()

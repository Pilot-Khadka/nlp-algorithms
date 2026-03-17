import math
import torch
import torch.nn as nn
import numpy as np

import matplotlib.pyplot as plt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
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


def example_positional():
    pe = PositionalEncoding(20, dropout=0)
    y = pe.forward(torch.zeros(1, 100, 20))  # shape (1, 100, 20)
    dims_to_plot = [4, 5, 6, 7]
    positions = np.arange(100)
    y_np = y.detach().numpy()

    plt.figure(figsize=(12, 6))
    for dim in dims_to_plot:
        plt.plot(positions, y_np[0, :, dim], label=f"dim {dim}")

    plt.xlabel("Position")
    plt.ylabel("Encoding Value")
    plt.title("Sinusoidal Positional Encoding (PyTorch → NumPy → Matplotlib)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    example_positional()

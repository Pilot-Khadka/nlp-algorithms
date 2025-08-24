import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Args:
        d_model (int): The embedding dimension.
        max_len (int): The maximum length of the sequence.
        method (str): The type of positional encoding to use.
                      Can be 'sinusoidal' or 'learned'.
    """

    def __init__(self, d_model, max_len=5000, method="sinusoidal"):
        super().__init__()
        self.d_model = d_model
        self.method = method

        if self.method == "sinusoidal":
            # sinusodial positional encoding as described in "Attention Is All You Need"
            self.pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )

            self.pe[:, 0::2] = torch.sin(position * div_term)
            self.pe[:, 1::2] = torch.cos(position * div_term)

            # register buffer to prevent the tensor from being considered a model parameter
            self.register_buffer("pe", self.pe.unsqueeze(0))

        elif self.method == "learned":
            # learned positional embeddings, similar to what BERT uses
            self.pos_embedding = nn.Embedding(max_len, d_model)

        else:
            raise ValueError("Method must be 'sinusoidal' or 'learned'")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: The input tensor with positional encoding added.
        """
        seq_len = x.size(1)

        if self.method == "sinusoidal":
            return x + self.pe[:, :seq_len, :]

        elif self.method == "learned":
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
            return x + self.pos_embedding(positions)


if __name__ == "__main__":
    d_model = 512
    seq_len = 10
    batch_size = 2

    pos_enc = PositionalEncoding(d_model, max_len=1000)
    x = torch.randn(batch_size, seq_len, d_model)
    output = pos_enc(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

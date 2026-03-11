import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model
from .qrnn_layer import QRNNLayer


@register_model("qrnn")
class QRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        kernel_size=2,
        batch_first=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            layers.append(QRNNLayer(in_dim, hidden_dim, kernel_size))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, c0=None):
        """
        x: (B, T, D) if batch_first else (T, B, D)
        c0: (num_layers, B, H) or None

        returns:
            output: (B, T, H)
            h_n: (num_layers, B, H)
        """
        if not self.batch_first:
            # Convert to batch_first temporarily
            x = x.transpose(0, 1)

        B, T, _ = x.shape

        if c0 is None:
            c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)

        c_n = []

        out = x
        for layer_idx, layer in enumerate(self.layers):
            h_prev = c0[layer_idx]
            out, h_last = layer(out, hidden=h_prev)
            c_n.append(h_last)

        # (num_layers, B, H)
        h_n = torch.stack(c_n, dim=0)

        if not self.batch_first:
            # (T, B, H)
            out = out.transpose(0, 1)

        return out, h_n

from typing import cast

import torch
import torch.nn as nn


from .qrnn_layer import QRNNLayer
from nlp_algorithms.engine.registry import register_model
from nlp_algorithms.lstm.locked_dropout import LockedDropout


@register_model("qrnn")
class QRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers=1,
        kernel_size=2,
        dropout=0.0,
        hidden_dropout=0.0,
        batch_first=True,
        use_locked_dropout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.use_locked_dropout = use_locked_dropout

        def make_dropout():
            return LockedDropout(dropout) if use_locked_dropout else nn.Dropout(dropout)

        self.layer_dropouts = nn.ModuleList(
            [make_dropout() for _ in range(num_layers - 1)]
        )
        self.output_dropout = make_dropout()

        self.layers = nn.ModuleList(
            [
                QRNNLayer(
                    input_dim if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size,
                    hidden_dropout=hidden_dropout if use_locked_dropout else 0.0,
                )
                for i in range(num_layers)
            ]
        )

    def _reset_dropout_masks(self):
        for dropout in self.layer_dropouts:
            if isinstance(dropout, LockedDropout):
                dropout.reset_mask()

        if isinstance(self.output_dropout, LockedDropout):
            self.output_dropout.reset_mask()

        for layer in self.layers:
            if isinstance(layer.hidden_dropout, LockedDropout):
                layer.hidden_dropout.reset_mask()

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

        if self.use_locked_dropout:
            self._reset_dropout_masks()

        c_n = []
        out = x
        for layer_idx, layer in enumerate(self.layers):
            h_prev = c0[layer_idx]
            out, h_last = layer(out, hidden=h_prev)
            c_n.append(h_last)

            if layer_idx < self.num_layers - 1:
                out = self.layer_dropouts[layer_idx](out)

        out = self.output_dropout(out)

        # (num_layers, B, H)
        h_n = torch.stack(c_n, dim=0)
        if not self.batch_first:
            # (T, B, H)
            out = out.transpose(0, 1)

        return out, h_n

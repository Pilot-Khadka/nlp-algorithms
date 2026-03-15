from typing import cast, Optional

import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model
from nlp_algorithms.lstm.locked_dropout import LockedDropout

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("gru")
class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=0.0,
        hidden_dropout=0.0,
        use_locked_dropout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.hidden_dropout = hidden_dropout
        self.use_locked_dropout = use_locked_dropout

        if use_locked_dropout:
            self.layer_dropouts = nn.ModuleList(
                [LockedDropout(hidden_dropout) for _ in range(num_layers)]
            )
            self.output_dropout = LockedDropout(dropout)
        else:
            self.layer_dropouts = nn.ModuleList(
                [nn.Dropout(hidden_dropout) for _ in range(num_layers)]
            )
            self.output_dropout = nn.Dropout(dropout)

        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()

        for layer in range(num_layers):
            input_size = input_dim if layer == 0 else hidden_dim
            H = hidden_dim
            # PyTorch GRU convention: rows are [r, z, n] gates
            self.w_ih.append(nn.Parameter(torch.empty(3 * H, input_size)))
            self.w_hh.append(nn.Parameter(torch.empty(3 * H, H)))
            self.b_ih.append(nn.Parameter(torch.zeros(3 * H)))
            self.b_hh.append(nn.Parameter(torch.zeros(3 * H)))

        self._initialize_weights()

    def _initialize_weights(self):
        for w_ih, w_hh in zip(self.w_ih, self.w_hh):
            nn.init.orthogonal_(w_ih)
            nn.init.orthogonal_(w_hh)

    def _reset_dropout_masks(self) -> None:
        if not self.use_locked_dropout:
            return
        for dropout in self.layer_dropouts:
            cast(LockedDropout, dropout).reset_mask()
        cast(LockedDropout, self.output_dropout).reset_mask()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.size()
        H = self.hidden_dim

        if hidden is None:
            h = [
                torch.zeros(batch_size, H, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h = [hidden[i] for i in range(self.num_layers)]

        self._reset_dropout_masks()

        layer_input = x
        for layer_idx in range(self.num_layers):
            w_ih = self.w_ih[layer_idx]
            w_hh = nn.functional.dropout(
                self.w_hh[layer_idx], p=self.hidden_dropout, training=self.training
            )
            b_ih = self.b_ih[layer_idx]
            b_hh = self.b_hh[layer_idx]

            x_proj = layer_input @ w_ih.t() + b_ih  # (batch, seq, 3H)
            outputs = torch.empty(
                batch_size, seq_len, H, device=x.device, dtype=x.dtype
            )

            for t in range(seq_len):
                h_proj = h[layer_idx] @ w_hh.t() + b_hh  # (batch, 3H)

                r_t = torch.sigmoid(x_proj[:, t, :H] + h_proj[:, :H])
                z_t = torch.sigmoid(x_proj[:, t, H : 2 * H] + h_proj[:, H : 2 * H])
                n_t = torch.tanh(x_proj[:, t, 2 * H :] + r_t * h_proj[:, 2 * H :])

                h[layer_idx] = (1 - z_t) * n_t + z_t * h[layer_idx]
                outputs[:, t, :] = h[layer_idx]

            if layer_idx < self.num_layers - 1:
                layer_input = self.layer_dropouts[layer_idx](outputs)
            else:
                layer_input = outputs

        output = self.output_dropout(layer_input)

        if not self.batch_first:
            output = output.transpose(0, 1).contiguous()

        h_n = torch.stack(h, dim=0)
        return output, h_n


if __name__ == "__main__":
    model = GRU(input_dim=10, hidden_dim=20, num_layers=1)
    print(model)

    x_dummy = torch.randn(3, 7, 10)  # (batch_size, seq_len, input_dim)
    output, h_n = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())
    output.mean().backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

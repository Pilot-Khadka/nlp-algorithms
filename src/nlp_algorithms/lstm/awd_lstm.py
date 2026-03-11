from typing import List, Tuple, Optional, cast

import torch
import torch.nn as nn


from nlp_algorithms.engine.registry import register_model
from .locked_dropout import LockedDropout


class WordDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask = x.new_empty(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)
        return x * mask


@register_model("awd-lstm")
class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        batch_first: bool = True,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.25,
        output_dropout: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layer_dropouts = nn.ModuleList(
            [LockedDropout(dropout) for _ in range(num_layers)]
        )
        self.recurrent_dropouts = nn.ModuleList(
            [LockedDropout(recurrent_dropout) for _ in range(num_layers)]
        )
        self.output_dropout = LockedDropout(output_dropout)

        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.bias = nn.ParameterList()

        for layer in range(self.num_layers):
            input_size = input_dim if layer == 0 else hidden_dim
            H = self.hidden_dim
            self.w_ih.append(nn.Parameter(torch.empty(input_size, 4 * H)))
            self.w_hh.append(nn.Parameter(torch.empty(H, 4 * H)))
            self.bias.append(nn.Parameter(torch.empty(4 * H)))

        self._initialize_weights()

    def _initialize_weights(self):
        init_range = 0.1
        H = self.hidden_dim
        for w_ih, w_hh, bias in zip(self.w_ih, self.w_hh, self.bias):
            nn.init.uniform_(w_ih, -init_range, init_range)
            nn.init.uniform_(w_hh, -init_range, init_range)
            nn.init.zeros_(bias)
            bias.data[H : 2 * H] = 1.0

    def _reset_dropout_masks(self) -> None:
        for dropout in self.layer_dropouts:
            cast(LockedDropout, dropout).reset_mask()
        for dropout in self.recurrent_dropouts:
            cast(LockedDropout, dropout).reset_mask()
        self.output_dropout.reset_mask()

    def _forward(
        self,
        x: torch.Tensor,
        hidden: List[torch.Tensor],
        cell: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        batch_size, seq_len, _ = x.size()
        H = self.hidden_dim
        L = self.num_layers

        h = [h_i.clone() for h_i in hidden]
        c = [c_i.clone() for c_i in cell]

        layer_input = x
        outputs_t = torch.empty_like(layer_input)
        for layer_idx in range(L):
            w_ih = self.w_ih[layer_idx]
            w_hh = self.w_hh[layer_idx]
            bias = self.bias[layer_idx]
            recurrent_dropout = self.recurrent_dropouts[layer_idx]

            x_proj = layer_input @ w_ih

            outputs_t = torch.empty(
                batch_size, seq_len, H, device=x.device, dtype=x.dtype
            )

            for t in range(seq_len):
                # first call creates mask, subsequent calls reuse it
                h_dropped = recurrent_dropout(h[layer_idx])

                gates = x_proj[:, t, :] + h_dropped @ w_hh + bias

                i = torch.sigmoid(gates[:, :H])
                f = torch.sigmoid(gates[:, H : 2 * H])
                g = torch.tanh(gates[:, 2 * H : 3 * H])
                o = torch.sigmoid(gates[:, 3 * H :])

                c[layer_idx] = f * c[layer_idx] + i * g
                h[layer_idx] = o * torch.tanh(c[layer_idx])

                outputs_t[:, t, :] = h[layer_idx]

            if layer_idx < L - 1:
                outputs_t = self.layer_dropouts[layer_idx](outputs_t)

            layer_input = outputs_t

        return outputs_t, h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not self.batch_first:
            x = x.transpose(0, 1)

        batch_size, seq_len, _ = x.size()
        H = self.hidden_dim

        if hidden is None:
            h = [
                torch.zeros(batch_size, H, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(batch_size, H, device=x.device, dtype=x.dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = hidden
            h = [h_0[i] for i in range(self.num_layers)]
            c = [c_0[i] for i in range(self.num_layers)]

        self._reset_dropout_masks()

        layer_input = x
        for layer_idx in range(self.num_layers):
            w_ih = self.w_ih[layer_idx]
            w_hh = self.w_hh[layer_idx]
            bias = self.bias[layer_idx]

            x_proj = layer_input @ w_ih
            outputs = torch.empty(
                batch_size, seq_len, H, device=x.device, dtype=x.dtype
            )

            for t in range(seq_len):
                h_dropped = self.recurrent_dropouts[layer_idx](h[layer_idx])
                gates = x_proj[:, t, :] + h_dropped @ w_hh + bias

                i = torch.sigmoid(gates[:, :H])
                f = torch.sigmoid(gates[:, H : 2 * H])
                g = torch.tanh(gates[:, 2 * H : 3 * H])
                o = torch.sigmoid(gates[:, 3 * H :])

                c[layer_idx] = f * c[layer_idx] + i * g
                h[layer_idx] = o * torch.tanh(c[layer_idx])
                outputs[:, t, :] = h[layer_idx]

            if layer_idx < self.num_layers - 1:
                layer_input = self.layer_dropouts[layer_idx](outputs)
            else:
                layer_input = outputs

        output = self.output_dropout(layer_input)

        if not self.batch_first:
            output = output.transpose(0, 1).contiguous()

        h_n = torch.stack(h, dim=0)
        c_n = torch.stack(c, dim=0)

        return output, (h_n, c_n)

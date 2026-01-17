from typing import List, Tuple, Optional, cast

import torch
import torch.nn as nn


from engine.registry import register_model
from a07_lstm.locked_dropout import LockedDropout

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


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
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        embed_dropout: float = 0.1,
        recurrent_dropout: float = 0.25,
        output_dropout: float = 0.5,
        weight_tying: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.weight_tying = weight_tying

        self.embedding = kwargs.get("embedding_layer", None)
        assert self.embedding is not None

        self.embed_dropout = WordDropout(embed_dropout)
        self.layer_dropouts = nn.ModuleList(
            [LockedDropout(dropout) for _ in range(num_layers)]
        )
        self.recurrent_dropouts = nn.ModuleList(
            [LockedDropout(recurrent_dropout) for _ in range(num_layers)]
        )
        self.output_dropout_module = LockedDropout(output_dropout)

        self._build_layers()
        self._setup_output_layers()
        self._initialize_weights()

    def _build_layers(self):
        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.bias = nn.ParameterList()

        for layer in range(self.num_layers):
            input_size = self.input_dim if layer == 0 else self.hidden_dim
            H = self.hidden_dim
            self.w_ih.append(nn.Parameter(torch.empty(input_size, 4 * H)))
            self.w_hh.append(nn.Parameter(torch.empty(H, 4 * H)))
            self.bias.append(nn.Parameter(torch.empty(4 * H)))

    def _setup_output_layers(self):
        assert self.embedding is not None
        if self.weight_tying:
            if self.hidden_dim == self.input_dim:
                self.proj = None
                self.fc = nn.Linear(self.hidden_dim, self.output_dim)
                self.fc.weight = self.embedding.weight
                print(f"[Weight Tying] Direct: hidden_dim={self.hidden_dim}")
            else:
                self.proj = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
                self.fc = nn.Linear(self.input_dim, self.output_dim)
                self.fc.weight = self.embedding.weight
                print(
                    f"[Weight Tying] With projection: {self.hidden_dim} -> {self.input_dim}"
                )
        else:
            self.proj = None
            self.fc = nn.Linear(self.hidden_dim, self.output_dim)
            print("[Weight Tying] Disabled")

    def _initialize_weights(self):
        init_range = 0.1

        assert self.embedding is not None
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)

        H = self.hidden_dim
        for w_ih, w_hh, bias in zip(self.w_ih, self.w_hh, self.bias):
            nn.init.uniform_(w_ih, -init_range, init_range)
            nn.init.uniform_(w_hh, -init_range, init_range)
            nn.init.zeros_(bias)
            bias.data[H : 2 * H] = 1.0  # forget gate
            bias.data[3 * H : 4 * H] = 0.0  # output gate

        if self.proj is not None:
            nn.init.uniform_(self.proj.weight, -init_range, init_range)

        nn.init.zeros_(self.fc.bias)

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        hidden = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        cell = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        return hidden, cell

    def _reset_dropout_masks(self) -> None:
        for dropout in self.layer_dropouts:
            cast(LockedDropout, dropout).reset_mask()
        for dropout in self.recurrent_dropouts:
            cast(LockedDropout, dropout).reset_mask()
        self.output_dropout_module.reset_mask()

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
        hidden: Optional[List[torch.Tensor]] = None,
        cell: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        batch_size, seq_len = x.shape

        self._reset_dropout_masks()

        assert self.embedding is not None
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)

        if hidden is not None and cell is not None:
            pass
        else:
            hidden, cell = self.init_hidden(batch_size, x.device, emb.dtype)

        lstm_out, new_hidden, new_cell = self._forward(emb, hidden, cell)
        lstm_out = self.output_dropout_module(lstm_out)

        if self.proj is not None:
            lstm_out = self.proj(lstm_out)

        output = self.fc(lstm_out)

        return output, (new_hidden, new_cell)

from typing import cast, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


from nlp_algorithms.engine.registry import register_model
from nlp_algorithms.lstm.locked_dropout import LockedDropout


class WeightDrop(nn.Module):
    def __init__(self, module: nn.RNNBase, weight_name: str, dropout: float):
        super().__init__()
        self.module = module
        self.weight_name = weight_name
        self.dropout = dropout

        w = module._parameters.pop(weight_name)
        module.register_parameter(f"{weight_name}_raw", nn.Parameter(w.data))

    def forward(self, *args, **kwargs) -> tuple[Tensor, Tensor]:
        raw = getattr(self.module, f"{self.weight_name}_raw")
        dropped = F.dropout(raw, p=self.dropout, training=self.training)
        object.__setattr__(self.module, self.weight_name, dropped)
        self.module._flat_weights = [
            getattr(self.module, name) for name in self.module._flat_weights_names
        ]
        return self.module(*args, **kwargs)


@register_model("gru", flags=["pytorch"])
class GRU(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        batch_first: bool = True,
        dropout: float = 0.0,
        hidden_dropout: float = 0.0,
        use_locked_dropout: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.use_locked_dropout = use_locked_dropout

        self.gru_layers = nn.ModuleList()
        for layer in range(num_layers):
            input_size = input_dim if layer == 0 else hidden_dim
            gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            if hidden_dropout > 0.0:
                gru = WeightDrop(gru, "weight_hh_l0", hidden_dropout)
            self.gru_layers.append(gru)

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

        self._reset_dropout_masks()

        h_list: list[Optional[torch.Tensor]] = (
            [hidden[i].unsqueeze(0) for i in range(self.num_layers)]
            if hidden is not None
            else [None] * self.num_layers
        )

        layer_input = x
        h_out: list[torch.Tensor] = []

        for layer_idx in range(self.num_layers):
            output, h_n = self.gru_layers[layer_idx](layer_input, h_list[layer_idx])
            h_out.append(h_n.squeeze(0))
            layer_input = (
                self.layer_dropouts[layer_idx](output)
                if layer_idx < self.num_layers - 1
                else output
            )

        output = self.output_dropout(layer_input)

        if not self.batch_first:
            output = output.transpose(0, 1).contiguous()

        return output, torch.stack(h_out, dim=0)


@register_model("gru", flags=["pytorch", "bidirectional"])
class BiGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, hidden=None):
        return self.gru(x, hidden)


if __name__ == "__main__":
    model = GRU(input_dim=10, hidden_dim=20, num_layers=1)
    print(model)

    x_dummy = torch.randn(3, 7, 10)
    output, h = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())

    output.mean().backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model
from nlp_algorithms.lstm.locked_dropout import LockedDropout


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("rnn")
class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=0.0,
        nonlinearity="tanh",
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.nonlinearity = nonlinearity
        self.dropout_layers = nn.ModuleList(
            [LockedDropout(p=dropout) for _ in range(num_layers - 1)]
        )
        self.gates_x = nn.ModuleList()
        self.gates_h = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.gates_x.append(nn.Linear(layer_input_dim, hidden_dim))
            self.gates_h.append(nn.Linear(hidden_dim, hidden_dim))
            setattr(self, f"weight_ih_l{layer}", self.gates_x[layer].weight)
            setattr(self, f"weight_hh_l{layer}", self.gates_h[layer].weight)
            setattr(self, f"bias_ih_l{layer}", self.gates_x[layer].bias)
            setattr(self, f"bias_hh_l{layer}", self.gates_h[layer].bias)

    def forward(self, x, hidden=None):
        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)
        x = x if self.batch_first else x.transpose(0, 1)
        act = torch.tanh if self.nonlinearity == "tanh" else torch.relu

        for dropout in self.dropout_layers:
            # pyrefly: ignore
            dropout.reset_mask()

        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h = [hidden[i] for i in range(self.num_layers)]

        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]
            for layer in range(self.num_layers):
                h[layer] = act(
                    self.gates_x[layer](layer_input) + self.gates_h[layer](h[layer])
                )
                if layer < self.num_layers - 1:
                    layer_input = self.dropout_layers[layer](h[layer])
                else:
                    layer_input = h[layer]
            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        h_n = torch.stack(h, dim=0)
        if not self.batch_first:
            outputs = outputs.transpose(0, 1).contiguous()
        return outputs, h_n

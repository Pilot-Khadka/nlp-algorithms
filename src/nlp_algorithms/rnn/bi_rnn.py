import torch
import torch.nn as nn


from nlp_algorithms.engine.registry import register_model


@register_model("rnn", flags=["bidirectional"])
class BidirectionalRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.bidirectional = True
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_layer = nn.Dropout(p=dropout)

        self.gates_x_fwd = nn.ModuleList()
        self.gates_h_fwd = nn.ModuleList()
        self.gates_x_bwd = nn.ModuleList()
        self.gates_h_bwd = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else 2 * hidden_dim
            self.gates_x_fwd.append(nn.Linear(layer_input_dim, hidden_dim))
            self.gates_h_fwd.append(nn.Linear(hidden_dim, hidden_dim))
            self.gates_x_bwd.append(nn.Linear(layer_input_dim, hidden_dim))
            self.gates_h_bwd.append(nn.Linear(hidden_dim, hidden_dim))

            setattr(self, f"weight_ih_l{layer}", self.gates_x_fwd[layer].weight)
            setattr(self, f"weight_hh_l{layer}", self.gates_h_fwd[layer].weight)
            setattr(self, f"bias_ih_l{layer}", self.gates_x_fwd[layer].bias)
            setattr(self, f"bias_hh_l{layer}", self.gates_h_fwd[layer].bias)
            setattr(self, f"weight_ih_l{layer}_reverse", self.gates_x_bwd[layer].weight)
            setattr(self, f"weight_hh_l{layer}_reverse", self.gates_h_bwd[layer].weight)
            setattr(self, f"bias_ih_l{layer}_reverse", self.gates_x_bwd[layer].bias)
            setattr(self, f"bias_hh_l{layer}_reverse", self.gates_h_bwd[layer].bias)

    def forward(self, x, hidden=None):
        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)
        x = x if self.batch_first else x.transpose(0, 1)

        if hidden is None:
            h_fwd = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
            h_bwd = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_fwd = [hidden[layer * 2] for layer in range(self.num_layers)]
            h_bwd = [hidden[layer * 2 + 1] for layer in range(self.num_layers)]

        layer_input = x
        for layer in range(self.num_layers):
            fwd_out = torch.empty(batch_size, seq_len, self.hidden_dim, device=x.device)
            bwd_out = torch.empty(batch_size, seq_len, self.hidden_dim, device=x.device)

            for t in range(seq_len):
                h_fwd[layer] = torch.tanh(
                    self.gates_x_fwd[layer](layer_input[:, t, :])
                    + self.gates_h_fwd[layer](h_fwd[layer])
                )
                fwd_out[:, t, :] = h_fwd[layer]

            for t in range(seq_len - 1, -1, -1):
                h_bwd[layer] = torch.tanh(
                    self.gates_x_bwd[layer](layer_input[:, t, :])
                    + self.gates_h_bwd[layer](h_bwd[layer])
                )
                bwd_out[:, t, :] = h_bwd[layer]

            layer_input = torch.cat((fwd_out, bwd_out), dim=2)
            if layer < self.num_layers - 1:
                layer_input = self.dropout_layer(layer_input)

        outputs = layer_input
        h_n = torch.stack([h for pair in zip(h_fwd, h_bwd) for h in pair], dim=0)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1).contiguous()

        return outputs, h_n


if __name__ == "__main__":
    pass

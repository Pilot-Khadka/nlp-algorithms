import torch
import torch.nn as nn

import torch.nn.functional as F


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_size = input_dim
        self.hidden_size = hidden_dim
        self.bias = bias

        # weight_ih: (3*hidden_size, input_size)
        # weight_hh: (3*hidden_size, hidden_size)
        self.weight_ih = nn.Parameter(torch.randn(3 * hidden_dim, input_dim) * 0.01)
        self.weight_hh = nn.Parameter(torch.randn(3 * hidden_dim, hidden_dim) * 0.01)

        if bias:
            self.bias_ih = nn.Parameter(torch.zeros(3 * hidden_dim))
            self.bias_hh = nn.Parameter(torch.zeros(3 * hidden_dim))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

    def forward(self, x_t, h_prev):
        hx = h_prev

        gates_x = F.linear(x_t, self.weight_ih, self.bias_ih)
        gates_h = F.linear(hx, self.weight_hh, self.bias_hh)

        x_r, x_z, x_n = gates_x.chunk(3, dim=1)
        h_r, h_z, h_n = gates_h.chunk(3, dim=1)
        r_t = torch.sigmoid(x_r + h_r)
        z_t = torch.sigmoid(x_z + h_z)
        n_t = torch.tanh(x_n + r_t * h_n)
        h_t = (1 - z_t) * n_t + z_t * hx
        return h_t


class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        batch_first=False,
        dropout=0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            cell = GRUCell(layer_input_dim, hidden_dim)
            self.layers.append(cell)

            setattr(self, f"weight_ih_l{layer}", cell.weight_ih)
            setattr(self, f"weight_hh_l{layer}", cell.weight_hh)
            setattr(self, f"bias_ih_l{layer}", cell.bias_ih)
            setattr(self, f"bias_hh_l{layer}", cell.bias_hh)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, hidden=None):
        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)
        x = x if self.batch_first else x.transpose(0, 1)

        if hidden is None:
            hidden = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_hidden = []

            for i, layer in enumerate(self.layers):
                h_prev = hidden[i]
                h_t = layer(x_t, h_prev)

                if i < self.num_layers - 1:
                    x_t = self.dropout(h_t)
                else:
                    x_t = h_t

                new_hidden.append(h_t)

            hidden = new_hidden
            outputs.append(x_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1).contiguous()

        return outputs, torch.stack(hidden)


if __name__ == "__main__":
    model = GRU(input_dim=10, hidden_dim=20, output_dim=5)
    print(model)

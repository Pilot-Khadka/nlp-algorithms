import torch
import torch.nn as nn


from engine.registry import register_model
from engine.model_factory import BaseModel


class GRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_layernorm=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        self.w_rz = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.u_rz = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)

        self.w_h = nn.Linear(input_dim, hidden_dim, bias=True)
        self.u_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if use_layernorm:
            self.ln_rz = nn.LayerNorm(2 * hidden_dim)
            self.ln_h = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, h_prev):
        rz_gates = self.w_rz(x_t) + self.u_rz(h_prev)
        if self.use_layernorm:
            rz_gates = self.ln_rz(rz_gates)

        r_t, z_t = torch.sigmoid(rz_gates).chunk(2, dim=1)
        h_candidate_part1 = self.w_h(x_t)
        h_candidate_part2 = r_t * self.u_h(h_prev)

        h_tilde = h_candidate_part1 + h_candidate_part2
        if self.use_layernorm:
            h_tilde = self.ln_h(h_tilde)

        h_tilde = torch.tanh(h_tilde)

        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_t


@register_model("gru")
class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = kwargs.get("embedding_layer", None)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(GRUCell(layer_input_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)

        batch_size, seq_len, _ = x.size()

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

        out = torch.cat(outputs, dim=1)
        return self.fc(out), torch.stack(hidden)


if __name__ == "__main__":
    model = GRU(input_dim=10, hidden_dim=20, output_dim=5)
    print(model)

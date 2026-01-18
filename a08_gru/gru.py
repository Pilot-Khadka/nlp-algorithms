import torch
import torch.nn as nn

from a08_gru.gru_cell import GRUCell
from engine.registry import register_model


@register_model("gru")
class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.0,
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

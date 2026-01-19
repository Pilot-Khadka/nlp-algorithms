import torch
import torch.nn as nn

from net_gru.gru_cell import GRUCell
from engine.registry import register_model


@register_model("gru", "bidirectional")
class BiGru(nn.Module):
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

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.forward_layers.append(GRUCell(layer_input_dim, hidden_dim))
            self.backward_layers.append(GRUCell(layer_input_dim, hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)

        batch_size, seq_len, _ = x.size()

        if hidden is None:
            fw_hidden = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
            bw_hidden = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            fw_hidden, bw_hidden = hidden

        fw_outputs = []
        bw_outputs = []

        # forward direction
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_hidden = []

            for i, layer in enumerate(self.forward_layers):
                h_t = layer(x_t, fw_hidden[i])
                x_t = self.dropout(h_t) if i < self.num_layers - 1 else h_t
                new_hidden.append(h_t)

            fw_hidden = new_hidden
            fw_outputs.append(x_t.unsqueeze(1))

        # backward direction
        for t in reversed(range(seq_len)):
            x_t = x[:, t, :]
            new_hidden = []

            for i, layer in enumerate(self.backward_layers):
                h_t = layer(x_t, bw_hidden[i])
                x_t = self.dropout(h_t) if i < self.num_layers - 1 else h_t
                new_hidden.append(h_t)

            bw_hidden = new_hidden
            bw_outputs.append(x_t.unsqueeze(1))

        bw_outputs.reverse()

        fw_out = torch.cat(fw_outputs, dim=1)
        bw_out = torch.cat(bw_outputs, dim=1)

        out = torch.cat([fw_out, bw_out], dim=-1)
        return self.fc(out), (torch.stack(fw_hidden), torch.stack(bw_hidden))


if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 16
    hidden_dim = 32
    output_dim = 8
    num_layers = 2

    model = BiGru(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=0.1,
    )

    x = torch.randn(batch_size, seq_len, input_dim)

    outputs, hidden = model(x)

    assert outputs.shape == (batch_size, seq_len, output_dim), (
        f"Unexpected output shape: {outputs.shape}"
    )

    fw_hidden, bw_hidden = hidden

    assert fw_hidden.shape == (num_layers, batch_size, hidden_dim), (
        f"Unexpected forward hidden shape: {fw_hidden.shape}"
    )

    assert bw_hidden.shape == (num_layers, batch_size, hidden_dim), (
        f"Unexpected backward hidden shape: {bw_hidden.shape}"
    )

    loss = outputs.mean()
    loss.backward()

    print("Output shape:", outputs.shape)
    print("Forward hidden shape:", fw_hidden.shape)
    print("Backward hidden shape:", bw_hidden.shape)

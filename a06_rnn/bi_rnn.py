import torch
import torch.nn as nn
import torch.nn.functional as F


from engine.registry import register_model


@register_model("rnn", "bidirectional")
class BidirectionalRNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.dropout = dropout
        self.embedding = kwargs.get("embedding_layer", None)

        self.forward_layers = nn.ModuleList()
        self.backward_layers = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else 2 * hidden_dim
            self.forward_layers.append(nn.Linear(in_dim + hidden_dim, hidden_dim))
            self.backward_layers.append(nn.Linear(in_dim + hidden_dim, hidden_dim))

        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def init_hidden(self, batch_size, device):
        return [
            (
                torch.zeros(batch_size, self.hidden_dim, device=device),
                torch.zeros(batch_size, self.hidden_dim, device=device),
            )
            for _ in range(self.num_layers)
        ]

    def forward(self, x, hidden=None):
        assert hidden is None
        if self.embedding:
            x = self.embedding(x)

        batch_size, seq_len, _ = x.size()

        hidden = self.init_hidden(batch_size, x.device)

        layer_input = x
        final_hidden = []

        last_forward_out = None
        last_backward_out = None

        for layer in range(self.num_layers):
            h_f, h_b = hidden[layer]

            forward_out = torch.empty(
                batch_size, seq_len, self.hidden_dim, device=x.device
            )
            backward_out = torch.empty(
                batch_size, seq_len, self.hidden_dim, device=x.device
            )

            forward_linear = self.forward_layers[layer]
            backward_linear = self.backward_layers[layer]

            # forward direction
            for t in range(seq_len):
                inp = torch.cat((layer_input[:, t, :], h_f), dim=1)
                h_f = torch.tanh(forward_linear(inp))
                forward_out[:, t, :] = h_f

            # backward direction
            for t in range(seq_len - 1, -1, -1):
                inp = torch.cat((layer_input[:, t, :], h_b), dim=1)
                h_b = torch.tanh(backward_linear(inp))
                backward_out[:, t, :] = h_b

            if self.dropout > 0 and layer < self.num_layers - 1:
                forward_out = F.dropout(
                    forward_out, p=self.dropout, training=self.training
                )
                backward_out = F.dropout(
                    backward_out, p=self.dropout, training=self.training
                )

            final_hidden.append((h_f, h_b))
            layer_input = torch.cat((forward_out, backward_out), dim=2)

            # save last layer outputs
            last_forward_out = forward_out
            last_backward_out = backward_out

        assert last_backward_out is not None
        assert last_forward_out is not None
        # bi_out = torch.cat((last_forward_out, last_backward_out), dim=2)
        # output_seq = self.fc(bi_out)

        h_f, h_b = final_hidden[-1]  # (batch, hidden_dim) each
        seq_repr = torch.cat((h_f, h_b), dim=1)  # (batch, 2 * hidden_dim)
        logits = self.fc(seq_repr)
        return logits, final_hidden
        # return output_seq, final_hidden


if __name__ == "__main__":
    pass

import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model


@register_model("lstm", flags=["bidirectional"])
class BiLSTM(nn.ModuleList):
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
        # used to determine if hidden should be halved
        self.bidirectional = True
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_layers - 1)]
        )

        self.gates_forward_x = nn.ModuleList()
        self.gates_forward_h = nn.ModuleList()
        self.gates_backward_x = nn.ModuleList()
        self.gates_backward_h = nn.ModuleList()

        for layer in range(num_layers):
            in_dim = input_dim if layer == 0 else 2 * hidden_dim

            self.gates_forward_x.append(nn.Linear(in_dim, 4 * hidden_dim))
            self.gates_forward_h.append(nn.Linear(hidden_dim, 4 * hidden_dim))

            self.gates_backward_x.append(nn.Linear(in_dim, 4 * hidden_dim))
            self.gates_backward_h.append(nn.Linear(hidden_dim, 4 * hidden_dim))

            setattr(self, f"weight_ih_l{layer}", self.gates_forward_x[layer].weight)
            setattr(self, f"weight_hh_l{layer}", self.gates_forward_h[layer].weight)
            setattr(self, f"bias_ih_l{layer}", self.gates_forward_x[layer].bias)
            setattr(self, f"bias_hh_l{layer}", self.gates_forward_h[layer].bias)

            setattr(
                self, f"weight_ih_reverse_l{layer}", self.gates_backward_x[layer].weight
            )
            setattr(
                self, f"weight_hh_reverse_l{layer}", self.gates_backward_h[layer].weight
            )
            setattr(
                self, f"bias_ih_reverse_l{layer}", self.gates_backward_x[layer].bias
            )
            setattr(
                self, f"bias_hh_reverse_l{layer}", self.gates_backward_h[layer].bias
            )

    def forward(self, x, hidden=None):
        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)
        x = x if self.batch_first else x.transpose(0, 1)

        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers * 2)
            ]  # forward + backward
            c = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers * 2)
            ]
        else:
            h_0, c_0 = hidden  # (num_layers*2, B, H)
            h = [h_0[i] for i in range(self.num_layers * 2)]
            c = [c_0[i] for i in range(self.num_layers * 2)]

        outputs = x

        for layer in range(self.num_layers):
            layer_input = outputs

            out_f_list = []
            out_b_list = []

            # forward
            h_f = h[layer * 2]
            c_f = c[layer * 2]

            for t in range(seq_len):
                gates = self.gates_forward_x[layer](
                    layer_input[:, t, :]
                ) + self.gates_forward_h[layer](h_f)
                f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)

                f_t = torch.sigmoid(f_t)
                i_t = torch.sigmoid(i_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                c_f = f_t * c_f + i_t * g_t
                h_f = o_t * torch.tanh(c_f)

                out_f_list.append(h_f.unsqueeze(1))

            # backward
            h_b = h[layer * 2 + 1]
            c_b = c[layer * 2 + 1]

            for t in reversed(range(seq_len)):
                gates = self.gates_backward_x[layer](
                    layer_input[:, t, :]
                ) + self.gates_backward_h[layer](h_b)
                f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)

                f_t = torch.sigmoid(f_t)
                i_t = torch.sigmoid(i_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                c_b = f_t * c_b + i_t * g_t
                h_b = o_t * torch.tanh(c_b)

                out_b_list.append(h_b.unsqueeze(1))

            out_b_list.reverse()

            out_f = torch.cat(out_f_list, dim=1)
            out_b = torch.cat(out_b_list, dim=1)

            outputs = torch.cat([out_f, out_b], dim=2)  # (B, T, 2H)

            h[layer * 2] = h_f
            c[layer * 2] = c_f
            h[layer * 2 + 1] = h_b
            c[layer * 2 + 1] = c_b

            if layer < self.num_layers - 1:
                layer_input = self.dropout_layers[layer](outputs)
            else:
                layer_input = h[layer]

        if not self.batch_first:
            outputs = outputs.transpose(0, 1).contiguous()

        h_n = torch.stack(h, dim=0)  # (num_layers*2, B, H)
        c_n = torch.stack(c, dim=0)
        return outputs, (h_n, c_n)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 6
    input_dim = 10
    hidden_dim = 8
    output_dim = 5

    model = BiLSTM(
        input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=1
    )
    model.train()

    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)

    output = model(x)
    print("Forward pass output shape:", output.shape)

    target = torch.randn(batch_size, seq_len, output_dim)

    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print("Loss:", loss.item())

    loss.backward()
    print("Backward pass completed.")

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(
                f"{name} grad mean: {param.grad.mean():.6f}, std: {param.grad.std():.6f}"
            )
        else:
            print(f"{name} has no gradient!")

import torch
import torch.nn as nn

from engine.registry import register_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("lstm")
class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = kwargs.get("embedding_layer", None)

        self.gates_forward_x = nn.ModuleList()
        self.gates_forward_h = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.gates_forward_x.append(nn.Linear(layer_input_dim, 4 * hidden_dim))
            self.gates_forward_h.append(nn.Linear(hidden_dim, 4 * hidden_dim))

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)

        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h = list(hidden)

        c = [
            torch.zeros(batch_size, self.hidden_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []
        for t in range(seq_len):
            layer_input = x[:, t, :]

            for layer in range(self.num_layers):
                all_gates = self.gates_forward_x[layer](
                    layer_input
                ) + self.gates_forward_h[layer](h[layer])

                f_t, i_t, g_t, o_t = all_gates.chunk(4, dim=1)

                f_t = torch.sigmoid(f_t)
                i_t = torch.sigmoid(i_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                c[layer] = f_t * c[layer] + i_t * g_t
                h[layer] = o_t * torch.tanh(c[layer])

                layer_input = h[layer]

            outputs.append(h[-1].unsqueeze(1))  # (B, 1, H)

        outputs = torch.cat(outputs, dim=1)
        output_seq = self.fc(outputs)

        return output_seq


if __name__ == "__main__":
    model = LSTM(input_dim=10, hidden_dim=20, output_dim=5, num_layers=1)
    print(model)

    x_dummy = torch.randn(3, 7, 10)  # (batch_size, seq_len, input_dim)
    output = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())
    output.mean().backward()  # dummy loss

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

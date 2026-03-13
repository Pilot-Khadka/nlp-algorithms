import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("lstm")
class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        dropout=0.0,
        use_locked_dropout: bool = False,
        proj_size=None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.use_locked_dropout = use_locked_dropout
        self.dropout_p = dropout

        self.proj_size = proj_size if proj_size is not None else hidden_dim
        self.dropout_layers = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_layers - 1)]
        )

        self.gates_forward_x = nn.ModuleList()
        self.gates_forward_h = nn.ModuleList()
        self.proj_layers = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else hidden_dim
            self.gates_forward_x.append(nn.Linear(layer_input_dim, 4 * hidden_dim))
            self.gates_forward_h.append(nn.Linear(hidden_dim, 4 * hidden_dim))

            if self.proj_size != hidden_dim:
                proj = nn.Linear(hidden_dim, self.proj_size, bias=False)
            else:
                proj = nn.Identity()

            self.proj_layers.append(proj)

            setattr(self, f"weight_ih_l{layer}", self.gates_forward_x[layer].weight)
            setattr(self, f"weight_hh_l{layer}", self.gates_forward_h[layer].weight)
            setattr(self, f"bias_ih_l{layer}", self.gates_forward_x[layer].bias)
            setattr(self, f"bias_hh_l{layer}", self.gates_forward_h[layer].bias)

            if self.proj_size != hidden_dim:
                setattr(self, f"weight_hr_l{layer}", proj.weight)

    def forward(self, x, hidden=None):
        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)
        x = x if self.batch_first else x.transpose(0, 1)

        if hidden is None:
            h = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
            c = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_0, c_0 = hidden  # each: (num_layers, batch, H)
            h = [h_0[i] for i in range(self.num_layers)]  # list of (batch, H)
            c = [c_0[i] for i in range(self.num_layers)]

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
                raw_h_t = o_t * torch.tanh(c[layer])

                h_t = self.proj_layers[layer](raw_h_t)
                h[layer] = h_t

                if layer < self.num_layers - 1:
                    layer_input = self.dropout_layers[layer](h_t)
                else:
                    layer_input = h[layer]

            outputs.append(h[-1].unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        h_n = torch.stack(h, dim=0)  # (num_layers, batch, H)
        c_n = torch.stack(c, dim=0)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1).contiguous()

        return outputs, (h_n, c_n)


if __name__ == "__main__":
    model = LSTM(input_dim=10, hidden_dim=20, num_layers=1)
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

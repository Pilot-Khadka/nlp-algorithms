import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("gru", flags=["bidirectional"])
class BiGRU(nn.Module):
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

        self.gates_forward_x = nn.ModuleList()
        self.gates_forward_h = nn.ModuleList()
        self.gates_backward_x = nn.ModuleList()
        self.gates_backward_h = nn.ModuleList()

        for layer in range(num_layers):
            layer_input_dim = input_dim if layer == 0 else 2 * hidden_dim
            self.gates_forward_x.append(nn.Linear(layer_input_dim, 3 * hidden_dim))
            self.gates_forward_h.append(nn.Linear(hidden_dim, 3 * hidden_dim))
            self.gates_backward_x.append(nn.Linear(layer_input_dim, 3 * hidden_dim))
            self.gates_backward_h.append(nn.Linear(hidden_dim, 3 * hidden_dim))

            setattr(self, f"weight_ih_l{layer}", self.gates_forward_x[layer].weight)
            setattr(self, f"weight_hh_l{layer}", self.gates_forward_h[layer].weight)
            setattr(self, f"bias_ih_l{layer}", self.gates_forward_x[layer].bias)
            setattr(self, f"bias_hh_l{layer}", self.gates_forward_h[layer].bias)

            setattr(
                self, f"weight_ih_l{layer}_reverse", self.gates_backward_x[layer].weight
            )
            setattr(
                self, f"weight_hh_l{layer}_reverse", self.gates_backward_h[layer].weight
            )
            setattr(
                self, f"bias_ih_l{layer}_reverse", self.gates_backward_x[layer].bias
            )
            setattr(
                self, f"bias_hh_l{layer}_reverse", self.gates_backward_h[layer].bias
            )

    def _gru_step(self, layer, x_t, h, gates_x, gates_h):
        r_x, z_x, n_x = gates_x[layer](x_t).chunk(3, dim=1)
        r_h, z_h, n_h = gates_h[layer](h).chunk(3, dim=1)

        r_t = torch.sigmoid(r_x + r_h)
        z_t = torch.sigmoid(z_x + z_h)
        n_t = torch.tanh(n_x + r_t * n_h)

        return (1 - z_t) * n_t + z_t * h

    def forward(self, x, hidden=None):
        batch_size = x.size(0) if self.batch_first else x.size(1)
        seq_len = x.size(1) if self.batch_first else x.size(0)
        x = x if self.batch_first else x.transpose(0, 1)

        if hidden is None:
            h_fw = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
            h_bw = [
                torch.zeros(batch_size, self.hidden_dim, device=x.device)
                for _ in range(self.num_layers)
            ]
        else:
            h_fw = [hidden[0][i] for i in range(self.num_layers)]
            h_bw = [hidden[1][i] for i in range(self.num_layers)]

        layer_input = x
        for layer in range(self.num_layers):
            fw_outputs = []
            bw_outputs = []

            for t in range(seq_len):
                h_fw[layer] = self._gru_step(
                    layer,
                    layer_input[:, t, :],
                    h_fw[layer],
                    self.gates_forward_x,
                    self.gates_forward_h,
                )
                fw_outputs.append(h_fw[layer].unsqueeze(1))

            for t in reversed(range(seq_len)):
                h_bw[layer] = self._gru_step(
                    layer,
                    layer_input[:, t, :],
                    h_bw[layer],
                    self.gates_backward_x,
                    self.gates_backward_h,
                )
                bw_outputs.append(h_bw[layer].unsqueeze(1))

            bw_outputs.reverse()

            out_f = torch.cat(fw_outputs, dim=1)
            out_b = torch.cat(bw_outputs, dim=1)

            outputs = torch.cat([out_f, out_b], dim=2)  # (B, T, 2H)
            # combined = torch.cat(
            #    [torch.cat(fw_outputs, dim=1), torch.cat(bw_outputs, dim=1)], dim=-1
            # )

            if layer < self.num_layers - 1:
                layer_input = self.dropout_layer(outputs)
            else:
                layer_input = outputs

        outputs = layer_input
        h_fw_n = torch.stack(h_fw, dim=0)
        h_bw_n = torch.stack(h_bw, dim=0)

        if not self.batch_first:
            outputs = outputs.transpose(0, 1).contiguous()

        return outputs, (h_fw_n, h_bw_n)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 10
    input_dim = 16
    hidden_dim = 32
    num_layers = 2

    model = BiGRU(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
    )

    x = torch.randn(batch_size, seq_len, input_dim)
    outputs, (h_fw, h_bw) = model(x)

    print("Output shape:", outputs.shape)
    print("Any NaNs in output?", torch.isnan(outputs).any().item())
    print("Output max abs:", outputs.abs().max().item())
    assert outputs.shape == (batch_size, seq_len, 2 * hidden_dim)
    assert h_fw.shape == (num_layers, batch_size, hidden_dim)
    assert h_bw.shape == (num_layers, batch_size, hidden_dim)

    outputs.mean().backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

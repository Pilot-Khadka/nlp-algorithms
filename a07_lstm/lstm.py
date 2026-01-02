from typing import List, Tuple


import torch
import torch.nn as nn


from engine.registry import register_model
from engine.model_factory import BaseModel

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("lstm")
class LSTM(BaseModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        use_compile=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = kwargs.get("embedding_layer", None)
        self.dropout_prob = kwargs.get("dropout", None)
        if self.dropout_prob is not None:
            self.dropout = nn.Dropout(self.dropout_prob)
        else:
            self.dropout = None

        if self.embedding is not None:
            self.embedding.weight.requires_grad = False
            self.input_dim = self.embedding.embedding_dim
        else:
            self.input_dim = input_dim

        H = hidden_dim

        self.w_ih = nn.ParameterList()  # input-to-hidden
        self.w_hh = nn.ParameterList()  # hidden-to-hidden
        self.bias = nn.ParameterList()

        for layer in range(num_layers):
            layer_input = self.input_dim if layer == 0 else H
            self.w_ih.append(nn.Parameter(torch.empty(layer_input, 4 * H)))
            self.w_hh.append(nn.Parameter(torch.empty(H, 4 * H)))
            self.bias.append(nn.Parameter(torch.zeros(4 * H)))

        self.fc = nn.Linear(hidden_dim, output_dim)

        if self.input_dim == self.hidden_dim and self.embedding is not None:
            self.fc.weight = self.embedding.weight
            print("Weight Tying Enabled")

        self._use_compile = use_compile
        self._compiled = False

    def _lstm_forward_impl(
        self,
        x: torch.Tensor,
        h_list: list[torch.Tensor],
        c_list: list[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        H = self.hidden_dim
        L = self.num_layers

        h = list(h_list)
        c = list(c_list)

        outputs = torch.empty(batch_size, seq_len, H, device=x.device, dtype=x.dtype)

        x_proj_0 = x @ self.w_ih[0]  # [batch, seq_len, 4*H]

        w_hh_0 = self.w_hh[0]
        bias_0 = self.bias[0]

        w_ih_layers = [self.w_ih[l] for l in range(1, L)]
        w_hh_layers = [self.w_hh[l] for l in range(1, L)]
        bias_layers = [self.bias[l] for l in range(1, L)]

        for t in range(seq_len):
            gates = x_proj_0[:, t, :] + h[0] @ w_hh_0 + bias_0
            ifo = torch.sigmoid(gates[:, : 3 * H])
            g = torch.tanh(gates[:, 3 * H :])
            i, f, o = ifo.chunk(3, dim=1)

            c[0] = f * c[0] + i * g
            h[0] = o * torch.tanh(c[0])

            x_t = h[0]
            if self.dropout is not None and L > 1:
                x_t = self.dropout(x_t)

            # remaining layers
            for layer_idx in range(L - 1):
                gates = (
                    x_t @ w_ih_layers[layer_idx]
                    + h[layer_idx + 1] @ w_hh_layers[layer_idx]
                    + bias_layers[layer_idx]
                )

                ifo = torch.sigmoid(gates[:, : 3 * H])
                g = torch.tanh(gates[:, 3 * H :])
                i, f, o = ifo.chunk(3, dim=1)

                c[layer_idx + 1] = f * c[layer_idx + 1] + i * g
                h[layer_idx + 1] = o * torch.tanh(c[layer_idx + 1])

                x_t = h[layer_idx + 1]
                if self.dropout is not None and layer_idx < L - 2:
                    x_t = self.dropout(x_t)

            outputs[:, t, :] = x_t

        return outputs

    def _compile(self):
        if self._use_compile and not self._compiled:
            self._lstm_forward_impl = torch.compile(
                self._lstm_forward_impl,
                mode="reduce-overhead",
            )
            self._compiled = True

    def forward(self, x, hidden=None, cell=None):
        if self.embedding:
            x = self.embedding(x)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = x.float()

        batch_size = x.size(0)
        device = x.device
        dtype = x.dtype

        if hidden is None:
            h_list = [
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            h_list = [hidden[l] for l in range(self.num_layers)]

        if cell is None:
            c_list = [
                torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
                for _ in range(self.num_layers)
            ]
        else:
            c_list = [cell[l] for l in range(self.num_layers)]

        self._compile()
        lstm_out = self._lstm_forward_impl(x, h_list, c_list)
        return self.fc(lstm_out)


class LSTMNaive(BaseModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_size = input_dim + hidden_dim
        self.embedding = kwargs.get("embedding_layer", None)
        self.forget_gate = nn.Linear(input_size, hidden_dim)
        self.input_gate1 = nn.Linear(input_size, hidden_dim)
        self.input_gate2 = nn.Linear(input_size, hidden_dim)
        self.output_gate = nn.Linear(input_size, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h_t = hidden
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t], dim=1)

            f_t = self.sigmoid(self.forget_gate(combined))
            i_t = self.sigmoid(self.input_gate1(combined))
            g_t = self.tanh(self.input_gate2(combined))
            o_t = self.sigmoid(self.output_gate(combined))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            outputs.append(h_t.unsqueeze(1))  # (B, 1, H)

        out = torch.cat(outputs, dim=1)
        return self.fc(out) if self.fc else out


if __name__ == "__main__":
    model = LSTM(input_dim=10, hidden_dim=20, output_dim=5)
    print(model)

    x_dummy = torch.randn(3, 7, 10)  # (batch_size, seq_len, embedding_dim)
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

    x = torch.randn(100, 7, 10)
    y = (x.sum(dim=(1, 2)) > 0).long()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

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
        if self.embedding is not None:
            self.embedding.weight.requires_grad = False
            self.input_dim = self.embedding.embedding_dim
        else:
            self.input_dim = input_dim

        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()

        for layer in range(num_layers):
            input_size = self.input_dim if layer == 0 else hidden_dim
            self.w_ih.append(nn.Parameter(torch.empty(4 * hidden_dim, input_size)))
            self.w_hh.append(nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim)))
            self.b_ih.append(nn.Parameter(torch.empty(4 * hidden_dim)))
            self.b_hh.append(nn.Parameter(torch.empty(4 * hidden_dim)))

        self.fc = nn.Linear(hidden_dim, output_dim)
        self._reset_parameters()

        self._use_compile = use_compile
        self._compiled = False

    def _reset_parameters(self):
        stdv = 1.0 / (self.hidden_dim**0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def _lstm_forward_impl(
        self, x: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        outputs = torch.empty(
            batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype
        )

        h_layers = list(hidden.unbind(0))
        c_layers = list(cell.unbind(0))

        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer in range(self.num_layers):
                h_t = h_layers[layer]
                c_t = c_layers[layer]

                gates = torch.addmm(
                    self.b_ih[layer], x_t, self.w_ih[layer].t()
                ) + torch.addmm(self.b_hh[layer], h_t, self.w_hh[layer].t())

                i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)

                i_t = torch.sigmoid(i_t)
                f_t = torch.sigmoid(f_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

                h_layers[layer] = h_t
                c_layers[layer] = c_t
                x_t = h_t

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

        if hidden is None:
            hidden = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )
        if cell is None:
            cell = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )

        # lazy compile on first forward pass
        self._compile()

        lstm_out = self._lstm_forward_impl(x, hidden, cell)

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

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.registry import register_model
from engine.model_factory import BaseModel


@register_model("lstm", "bidirectional")
class BiLSTM(BaseModel):
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
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = kwargs.get("embedding_layer", None)

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

        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        assert hidden is None
        if self.embedding:
            x = self.embedding(x)

        batch_size, seq_len, _ = x.size()

        for layer in range(self.num_layers):
            # project inputs for this layer
            x_proj_f = self.gates_forward_x[layer](x)
            x_proj_b = self.gates_backward_x[layer](x)

            h_t_f = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c_t_f = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            outputs_f = []

            for t in range(seq_len):
                gates = x_proj_f[:, t, :] + self.gates_forward_h[layer](h_t_f)
                f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)

                f_t = torch.sigmoid(f_t)
                i_t = torch.sigmoid(i_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                c_t_f = f_t * c_t_f + i_t * g_t
                h_t_f = o_t * torch.tanh(c_t_f)
                outputs_f.append(h_t_f.unsqueeze(1))

            h_t_b = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            c_t_b = torch.zeros(batch_size, self.hidden_dim, device=x.device)
            outputs_b = []

            for t in range(seq_len - 1, -1, -1):
                gates = x_proj_b[:, t, :] + self.gates_backward_h[layer](h_t_b)
                f_t, i_t, g_t, o_t = gates.chunk(4, dim=1)

                f_t = torch.sigmoid(f_t)
                i_t = torch.sigmoid(i_t)
                g_t = torch.tanh(g_t)
                o_t = torch.sigmoid(o_t)

                c_t_b = f_t * c_t_b + i_t * g_t
                h_t_b = o_t * torch.tanh(c_t_b)
                outputs_b.append(h_t_b.unsqueeze(1))

            outputs_b.reverse()

            outputs_f = torch.cat(outputs_f, dim=1)
            outputs_b = torch.cat(outputs_b, dim=1)

            x = torch.cat([outputs_f, outputs_b], dim=2)  # input to next layer

            if self.dropout > 0 and layer < self.num_layers - 1:
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.fc(x)


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

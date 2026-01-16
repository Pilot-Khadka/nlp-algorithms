import torch
import torch.nn as nn

from engine.registry import register_model
from engine.model_factory import BaseModel


@register_model("lstm", "bidirectional")
class BiLSTM(BaseModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = kwargs.get("embedding_layer", None)

        self.gates_forward_x = nn.Linear(input_dim, 4 * hidden_dim)
        self.gates_forward_h = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.gates_backward_x = nn.Linear(input_dim, 4 * hidden_dim)
        self.gates_backward_h = nn.Linear(hidden_dim, 4 * hidden_dim)

        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)

        batch_size, seq_len, _ = x.size()

        # precalculation, project all timesteps for x at once
        x_projections_f = self.gates_forward_x(x)
        x_projections_b = self.gates_backward_x(x)

        h_t_f = (
            torch.zeros(batch_size, self.hidden_dim, device=x.device)
            if hidden is None
            else hidden
        )
        c_t_f = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs_f = []

        for t in range(seq_len):
            all_gates = x_projections_f[:, t, :] + self.gates_forward_h(h_t_f)

            f_t, i_t, g_t, o_t = all_gates.chunk(4, dim=1)

            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t_f = f_t * c_t_f + i_t * g_t
            h_t_f = o_t * torch.tanh(c_t_f)
            outputs_f.append(h_t_f.unsqueeze(1))  # (B, 1, H)

        # backward direction
        h_t_b = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t_b = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs_b = []

        for t in range(seq_len - 1, -1, -1):
            all_gates = x_projections_b[:, t, :] + self.gates_backward_h(h_t_b)

            f_t, i_t, g_t, o_t = all_gates.chunk(4, dim=1)

            f_t = torch.sigmoid(f_t)
            i_t = torch.sigmoid(i_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t_b = f_t * c_t_b + i_t * g_t
            h_t_b = o_t * torch.tanh(c_t_b)
            # prepend to align with forward
            outputs_b.insert(0, h_t_b.unsqueeze(1))

        # concat forward and backward outputs
        outputs_f = torch.cat(outputs_f, dim=1)  # (B, T, H)
        outputs_b = torch.cat(outputs_b, dim=1)  # (B, T, H)
        bi_outputs = torch.cat([outputs_f, outputs_b], dim=2)  # (B, T, 2H)

        output_seq = self.fc(bi_outputs)  # (B, T, output_dim)
        return output_seq


if __name__ == "__main__":
    batch_size = 4
    seq_len = 6
    input_dim = 10
    hidden_dim = 8
    output_dim = 5

    model = BiLSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
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

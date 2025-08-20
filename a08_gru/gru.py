import torch
import torch.nn as nn
from engine.model_factory import BaseModel
from engine.registry import register_model

__register_model__ = True


@register_model("gru")
class GRU(BaseModel):
    def __init__(self, embedding_dim, hidden_dim, output_dim, **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim

        input_size = embedding_dim + hidden_dim
        self.embedding = kwargs.get("embedding_layer", None)

        # gates
        self.reset_gate = nn.Linear(input_size, hidden_dim)
        self.update_gate = nn.Linear(input_size, hidden_dim)
        self.candidate_hidden_gate = nn.Linear(input_size, hidden_dim)
        self.hidden_state = nn.Linear(hidden_dim, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        # activation fns
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        if hidden is None:
            h_t = torch.zeros(
                batch_size, self.hidden_dim, device=x.device, dtype=x.dtype
            )
        else:
            h_t = hidden

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t], dim=1)

            r_t = self.sigmoid(self.reset_gate(combined))
            z_t = self.sigmoid(self.update_gate(combined))

            # reset-gated hidden (Hadamard product, not matmul)
            h_reset = r_t * h_t

            # candidate from [x_t, r_t * h_{t-1}]
            cand_in = torch.cat([x_t, h_reset], dim=1)
            h_t_candidate = self.tanh(self.candidate_hidden_gate(cand_in))

            # use update gate to interpolate with previous state
            h_t = (1.0 - z_t) * h_t_candidate + z_t * h_t

            outputs.append(h_t.unsqueeze(1))  # (B, 1, H)

        out = torch.cat(outputs, dim=1)  # (B, T, H)
        return self.fc(out)


if __name__ == "__main__":
    model = GRU(embedding_dim=10, hidden_dim=20, output_dim=5)
    print(model)

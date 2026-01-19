import torch
import torch.nn as nn


class GRUCell(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        use_layernorm=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_layernorm = use_layernorm

        self.w_rz = nn.Linear(input_dim, 2 * hidden_dim, bias=True)
        self.u_rz = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)

        self.w_h = nn.Linear(input_dim, hidden_dim, bias=True)
        self.u_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        if use_layernorm:
            self.ln_rz = nn.LayerNorm(2 * hidden_dim)
            self.ln_h = nn.LayerNorm(hidden_dim)

    def forward(self, x_t, h_prev):
        rz_gates = self.w_rz(x_t) + self.u_rz(h_prev)
        if self.use_layernorm:
            rz_gates = self.ln_rz(rz_gates)

        r_t, z_t = torch.sigmoid(rz_gates).chunk(2, dim=1)
        h_candidate_part1 = self.w_h(x_t)
        h_candidate_part2 = r_t * self.u_h(h_prev)

        h_tilde = h_candidate_part1 + h_candidate_part2
        if self.use_layernorm:
            h_tilde = self.ln_h(h_tilde)

        h_tilde = torch.tanh(h_tilde)

        h_t = (1.0 - z_t) * h_prev + z_t * h_tilde
        return h_t

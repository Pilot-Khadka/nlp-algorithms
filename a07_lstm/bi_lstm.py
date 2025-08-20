import torch
import torch.nn as nn
from engine.model_factory import BaseModel
from engine.registry import register_model

__register_model__ = True


@register_model("bi_lstm")
class BiLSTM(BaseModel):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_size = embedding_dim + hidden_dim
        self.embedding = kwargs.get("embedding_layer", None)

        self.forget_gate_f = nn.Linear(input_size, hidden_dim)
        self.input_gate1_f = nn.Linear(input_size, hidden_dim)
        self.input_gate2_f = nn.Linear(input_size, hidden_dim)
        self.output_gate_f = nn.Linear(input_size, hidden_dim)

        self.forget_gate_b = nn.Linear(input_size, hidden_dim)
        self.input_gate1_b = nn.Linear(input_size, hidden_dim)
        self.input_gate2_b = nn.Linear(input_size, hidden_dim)
        self.output_gate_b = nn.Linear(input_size, hidden_dim)

        self.fc = nn.Linear(2 * hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        batch_size, seq_len, _ = x.size()

        # forward direction
        if hidden is None:
            h_t_f = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h_t_f = hidden
        c_t_f = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs_f = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t_f], dim=1)

            f_t = self.sigmoid(self.forget_gate_f(combined))
            i_t = self.sigmoid(self.input_gate1_f(combined))
            g_t = self.tanh(self.input_gate2_f(combined))
            o_t = self.sigmoid(self.output_gate_f(combined))

            c_t_f = f_t * c_t_f + i_t * g_t
            h_t_f = o_t * self.tanh(c_t_f)
            outputs_f.append(h_t_f.unsqueeze(1))  # (B, 1, H)

        # backward direction
        h_t_b = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        c_t_b = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        outputs_b = []

        for t in reversed(range(seq_len)):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t_b], dim=1)

            f_t = self.sigmoid(self.forget_gate_b(combined))
            i_t = self.sigmoid(self.input_gate1_b(combined))
            g_t = self.tanh(self.input_gate2_b(combined))
            o_t = self.sigmoid(self.output_gate_b(combined))

            c_t_b = f_t * c_t_b + i_t * g_t
            h_t_b = o_t * self.tanh(c_t_b)
            # prepend to align with forward
            outputs_b.insert(0, h_t_b.unsqueeze(1))

        # concat forward and backward outputs
        outputs_f = torch.cat(outputs_f, dim=1)  # (B, T, H)
        outputs_b = torch.cat(outputs_b, dim=1)  # (B, T, H)
        bi_outputs = torch.cat([outputs_f, outputs_b], dim=2)  # (B, T, 2H)

        output_seq = self.fc(bi_outputs)  # (B, T, output_dim)
        return output_seq

import torch
import torch.nn as nn

from engine.model_factory import BaseModel
from engine.registry import register_model

__register_model__ = True


@register_model("bidirectional_rnn")
class BidirectionalRNN(BaseModel):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        **kwargs,
    ):
        super(BidirectionalRNN, self).__init__()
        dropout_rate = kwargs.get("dropout_rate", 0.3)

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout_rate)

        self.i2h_f = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.i2h_b = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(2 * hidden_dim, output_dim)

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, input_seq, hidden=None):
        batch_size, seq_len, _ = input_seq.size()

        if hidden is None:
            h_f = torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)
            h_b = torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)

        outputs_f = []
        outputs_b = []

        # forward direction
        for t in range(seq_len):
            x_t = input_seq[:, t, :]
            h_f = self.activation(self.i2h_f(torch.cat((x_t, h_f), dim=1)))
            h_f = self.dropout(h_f)
            outputs_f.append(h_f)

        # backward direction
        for t in reversed(range(seq_len)):
            x_t = input_seq[:, t, :]
            h_b = self.activation(self.i2h_b(torch.cat((x_t, h_b), dim=1)))
            h_f = self.dropout(h_b)
            outputs_b.append(h_b)

        outputs_b.reverse()  # to match forward time order

        # concat forward and backward at each timestep
        bi_outputs = [
            torch.cat((f, b), dim=1) for f, b in zip(outputs_f, outputs_b)
        ]  # list of (batch, 2*hidden_dim)

        # (batch, seq_len, 2*hidden_dim)
        bi_outputs = torch.stack(bi_outputs, dim=1)
        output_seq = self.output_layer(bi_outputs)
        return output_seq


if __name__ == "__main__":
    pass

import torch
import torch.nn as nn
from engine.model_factory import BaseModel, create_model
from engine.registry import register_model

__register_model__ = True


@register_model("rnn")
class RNN(BaseModel):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        activation=nn.Tanh(),
        num_layers=2,
        **kwargs,
    ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # embed + hidden because
        # ht -> tanh(W h * xt + W h * ht-1 + b)
        self.i2h = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.h2h = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for i in range(self.num_layers)]
        )
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = activation

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_len, embedding_dim)
        # 32, 10, 768
        batch_size, seq_len, _ = x.size()

        # hidden state
        # 32, 768
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch_size, embedding_dim)
            combined = torch.cat([x_t, hidden], dim=1)
            h_t = self.activation_fn(self.i2h(combined))
            for i in range(self.num_layers):
                h_t = self.activation_fn(self.h2h[i](h_t))

            out_t = self.h2o(h_t)  # (batch_size, vocab_size)
            outputs.append(out_t.unsqueeze(1))  # (batch_size, 1, vocab_size)
            hidden = h_t  # carry hidden to next timestep

        # (batch_size, seq_len, vocab_size)
        outputs = torch.cat(outputs, dim=1)
        return outputs


def main():
    batch_size = 32
    seq_len = 10
    embedding_dim = 768
    vocab = 100

    model = create_model(
        model_type="rnn",
        vocab_size=vocab,
        embedding_dim=embedding_dim,
        hidden_dim=256,
        output_dim=vocab,
    )

    # token indices
    dummy_input = torch.randint(0, vocab, (batch_size, seq_len))

    output = model(dummy_input)
    print(output.shape)  # torch.Size([32, 10])


if __name__ == "__main__":
    main()

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        activation=nn.Tanh(),
    ):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        # embed + hidden because
        # ht -> tanh(W h * xt + W h * ht-1 + b)
        self.i2h = nn.Linear(embedding_dim + hidden_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)
        self.activation_fn = activation

    def init_hidden(self, batch_size, device):
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def forward(self, x):
        # x shape: (batch_size, seq_len, embedding_dim)
        # 32, 10, 768
        batch_size, seq_len, _ = x.size()

        # hidden state
        # 32, 768
        h_t = self.init_hidden(batch_size, x.device)

        for t in range(seq_len):
            # 32, 768
            x_t = x[:, t, :]  # (batch_size, embedding_dim)

            # (batch_size, embedding + hidden)
            # 32, 768 + 768
            combined = torch.cat([x_t, h_t], dim=1)
            h_t = self.activation_fn(self.i2h(combined))  # feedback loop

        output = self.h2o(h_t)  # (batch_size, output_dim)
        return output


def main():
    batch_size = 32
    seq_len = 10
    embedding_dim = 768
    rnn = RNN(embedding_dim=embedding_dim, hidden_dim=1000, output_dim=10)

    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    output = rnn(dummy_input)
    print(output.shape)  # torch.Size([32, 10])


if __name__ == "__main__":
    main()

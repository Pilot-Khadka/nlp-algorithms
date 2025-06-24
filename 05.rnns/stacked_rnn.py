import torch
import torch.nn as nn


class StackedRNN(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_stacks,
    ):
        super(StackedRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_stacks = num_stacks

        self.i2h = nn.Linear(self.embedding_dim + self.hidden_dim, self.hidden_dim)
        self.stacks = nn.ModuleList()

        # considering the hidden dim of stacks is also the same
        for i in range(self.num_stacks - 1):
            self.stacks.append(nn.Linear(self.hidden_dim * 2, self.hidden_dim))
        self.stacks.append(nn.Linear(self.hidden_dim * 2, self.output_dim))

        self.activation = nn.Tanh()

    def forward(self, input_seq, hidden=None):
        batch_size, seq_len, _ = input_seq.size()

        if hidden is None:
            hidden = [
                torch.zeros(batch_size, self.hidden_dim, device=input_seq.device)
                for _ in range(self.num_stacks)
            ]

        outputs = []
        for t in range(seq_len):
            x = input_seq[:, t, :]  # (batch, embedding_dim)

            h0_input = torch.cat((x, hidden[0]), dim=1)
            new_hidden = [self.activation(self.i2h(h0_input))]

            for i in range(1, self.num_stacks):
                h_input = torch.cat((new_hidden[i - 1], hidden[i]), dim=1)
                h_i = self.activation(self.stacks[i - 1](h_input))
                new_hidden.append(h_i)

            hidden = new_hidden
            outputs.append(hidden[-1])

        final_output = self.output_layer(outputs[-1])
        return final_output, hidden


if __name__ == "__main__":
    batch_size = 32
    seq_len = 10
    embedding_dim = 768
    rnn = StackedRNN(embedding_dim=embedding_dim, hidden_dim=1000, output_dim=10)

    dummy_input = torch.randn(batch_size, seq_len, embedding_dim)
    output = rnn(dummy_input)
    print(output.shape)  # torch.Size([32, 10])

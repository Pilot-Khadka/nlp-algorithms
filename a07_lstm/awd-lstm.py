import torch
import torch.nn as nn
from typing import Optional, Tuple, List


class VariationalDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def reset_mask(self, x: torch.Tensor, p: float = 0.5):
        if not self.training or p == 0:
            self.mask = None
        else:
            self.mask = x.new_empty(x.size(0), x.size(1)).bernoulli_(1 - p) / (1 - p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return x
        return x * self.mask


class AWDLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 400,
        hidden_dim: int = 1150,
        num_layers: int = 3,
        dropout_w: float = 0.5,
        dropout_i: float = 0.4,
        dropout_h: float = 0.25,
        dropout_o: float = 0.4,
        tie_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_w = dropout_w
        self.dropout_h = dropout_h

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lockdrop = nn.ModuleList(
            [VariationalDropout() for _ in range(num_layers - 1)]
        )
        self.dropout_i = nn.Dropout(dropout_i)
        self.dropout_o = nn.Dropout(dropout_o)

        self.w_ih = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(4 * hidden_dim, embedding_dim if l == 0 else hidden_dim)
                )
                for l in range(num_layers)
            ]
        )
        self.w_hh = nn.ParameterList(
            [
                nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
                for l in range(num_layers)
            ]
        )
        self.b_ih = nn.ParameterList(
            [nn.Parameter(torch.empty(4 * hidden_dim)) for l in range(num_layers)]
        )
        self.b_hh = nn.ParameterList(
            [nn.Parameter(torch.empty(4 * hidden_dim)) for l in range(num_layers)]
        )

        if tie_weights and embedding_dim != hidden_dim:
            self.proj = nn.Linear(hidden_dim, embedding_dim, bias=False)
        else:
            self.proj = None

        self.fc = nn.Linear(embedding_dim if tie_weights else hidden_dim, vocab_size)

        if tie_weights:
            self.fc.weight = self.embedding.weight

        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

            if self.proj is not None:
                nn.init.uniform_(self.proj.weight, -0.1, 0.1)

            if self.fc.bias is not None:
                nn.init.zeros_(self.fc.bias)

            for layer in range(self.num_layers):
                nn.init.uniform_(self.w_ih[layer], -0.1, 0.1)
                nn.init.uniform_(self.w_hh[layer], -0.1, 0.1)
                nn.init.zeros_(self.b_ih[layer])
                nn.init.zeros_(self.b_hh[layer])

                self.b_ih[layer][self.hidden_dim : 2 * self.hidden_dim].fill_(1)
                self.b_hh[layer][self.hidden_dim : 2 * self.hidden_dim].fill_(1)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]] = None,
    ) -> Tuple[
        torch.Tensor,
        Tuple[List[torch.Tensor], List[torch.Tensor]],
        torch.Tensor,
        torch.Tensor,
    ]:
        batch_size, seq_len = x.size()

        emb = self.dropout_i(self.embedding(x))

        if hidden is None:
            h = [
                emb.new_zeros(batch_size, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
            c = [
                emb.new_zeros(batch_size, self.hidden_dim)
                for _ in range(self.num_layers)
            ]
        else:
            h, c = hidden
            h = [h_i.detach() for h_i in h]
            c = [c_i.detach() for c_i in c]

        w_hh_dropped = [
            torch.nn.functional.dropout(
                self.w_hh[layer], p=self.dropout_w, training=self.training
            )
            for layer in range(self.num_layers)
        ]

        for layer in range(self.num_layers - 1):
            self.lockdrop[layer].reset_mask(
                emb.new_empty(batch_size, self.hidden_dim), self.dropout_h
            )

        raw_outputs = []
        outputs = []

        for t in range(seq_len):
            x_t = emb[:, t, :]

            for layer in range(self.num_layers):
                # gate order: (i, f, o, g) as in Merity et al.
                gates = torch.addmm(
                    self.b_ih[layer], x_t, self.w_ih[layer].t()
                ) + torch.addmm(self.b_hh[layer], h[layer], w_hh_dropped[layer].t())

                i, f, o, g = gates.chunk(4, dim=1)
                c[layer] = torch.sigmoid(f) * c[layer] + torch.sigmoid(i) * torch.tanh(
                    g
                )
                h[layer] = torch.sigmoid(o) * torch.tanh(c[layer])

                raw_outputs.append(h[layer]) if layer == self.num_layers - 1 else None

                if layer != self.num_layers - 1:
                    x_t = self.lockdrop[layer](h[layer])
                else:
                    x_t = h[layer]

            outputs.append(x_t)

        output = torch.stack(outputs, dim=1)
        raw_output = torch.stack(raw_outputs, dim=1)

        dropped_output = self.dropout_o(output)

        if self.proj is not None:
            dropped_output = self.proj(dropped_output)

        logits = self.fc(dropped_output)

        return logits, (h, c), dropped_output, raw_output


def awd_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    raw_outputs: torch.Tensor,
    dropped_outputs: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    AWD-LSTM loss with AR and TAR regularization

    AR uses dropped activations, TAR uses raw activations
    """
    ce_loss = nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1)
    )

    ar_loss = alpha * dropped_outputs.pow(2).mean()
    tar_loss = beta * (raw_outputs[:, 1:] - raw_outputs[:, :-1]).pow(2).mean()

    total_loss = ce_loss + ar_loss + tar_loss
    return total_loss, ar_loss, tar_loss


if __name__ == "__main__":
    model = AWDLSTM(
        vocab_size=10000,
        embedding_dim=400,
        hidden_dim=1150,
        num_layers=3,
        dropout_w=0.5,
        dropout_i=0.4,
        dropout_h=0.25,
        dropout_o=0.4,
    )

    x = torch.randint(0, 10000, (32, 70))
    targets = torch.randint(0, 10000, (32, 70))

    logits, hidden, dropped_out, raw_out = model(x)
    loss, ar, tar = awd_loss(logits, targets, raw_out, dropped_out)

    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.4f} (AR: {ar.item():.4f}, TAR: {tar.item():.4f})")

    with torch.no_grad():
        model.train()
        logits1, _, _, _ = model(x)
        logits2, _, _, _ = model(x)
        print(
            f"\nSame input -> different output (weight dropout): {not torch.allclose(logits1, logits2)}"
        )

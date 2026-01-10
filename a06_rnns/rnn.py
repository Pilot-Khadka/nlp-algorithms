from typing import Optional, Tuple

import torch
import torch.nn as nn

from engine.registry import register_model


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("rnn")
class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers=2,
        dropout=0.1,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.dropout_p = dropout

        self._hidden: Optional[torch.Tensor] = None

        self.embedding = kwargs.get("embedding_layer", None)
        if self.embedding is None:
            self.embedding = nn.Embedding(output_dim, input_dim)
            self.input_dim = input_dim
        else:
            self.input_dim = self.embedding.embedding_dim

        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()

        self.layer_norms = nn.ModuleList()

        for layer in range(num_layers):
            input_size = self.input_dim if layer == 0 else hidden_dim
            self.w_ih.append(nn.Parameter(torch.empty(hidden_dim, input_size)))
            self.w_hh.append(nn.Parameter(torch.empty(hidden_dim, hidden_dim)))
            self.b_ih.append(nn.Parameter(torch.empty(hidden_dim)))
            self.b_hh.append(nn.Parameter(torch.empty(hidden_dim)))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))

        self.input_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        self.h2o = nn.Linear(hidden_dim, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in range(self.num_layers):
            nn.init.xavier_uniform_(self.w_ih[layer])
            nn.init.orthogonal_(self.w_hh[layer])
            nn.init.zeros_(self.b_ih[layer])
            nn.init.zeros_(self.b_hh[layer])

        nn.init.xavier_uniform_(self.h2o.weight)
        if self.h2o.bias is not None:
            nn.init.zeros_(self.h2o.bias)

    @property
    def is_stateful(self) -> bool:
        return True

    def reset_state(self) -> None:
        self._hidden = None

    def get_state(self) -> Optional[torch.Tensor]:
        return self._hidden

    def set_state(self, state: Optional[torch.Tensor]) -> None:
        self._hidden = state

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        return torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

    def _rnn_forward_impl(
        self, x: torch.Tensor, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()

        outputs_list = []
        h_layers = list(hidden.unbind(0))

        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer in range(self.num_layers):
                h_new = torch.addmm(
                    self.b_ih[layer], x_t, self.w_ih[layer].t()
                ) + torch.addmm(self.b_hh[layer], h_layers[layer], self.w_hh[layer].t())

                h_new = self.layer_norms[layer](h_new)
                h_new = torch.tanh(h_new)
                h_layers[layer] = h_new

                if layer < self.num_layers - 1:
                    x_t = self.dropout(h_new)
                else:
                    x_t = h_new

            outputs_list.append(x_t)

        outputs = torch.stack(outputs_list, dim=1)
        final_hidden = torch.stack(h_layers, dim=0)
        return outputs, final_hidden

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = x.float()

        assert x.size(-1) == self.input_dim
        x = self.input_dropout(x)
        batch_size, seq_len, _ = x.size()

        if hidden is not None:
            pass
        elif self._hidden is not None:
            if self._hidden.size(1) == batch_size:
                hidden = self._hidden
            else:
                hidden = self.init_hidden(batch_size, x.device, x.dtype)
        else:
            hidden = self.init_hidden(batch_size, x.device, x.dtype)

        rnn_out, new_hidden = self._rnn_forward_impl(x, hidden)

        output = self.h2o(self.dropout(rnn_out))

        self._hidden = new_hidden.detach()
        return output, new_hidden


if __name__ == "__main__":
    import torch
    import torch.nn as nn

    class BaseModel(nn.Module):
        pass

    def register_model(name, *flags):
        def decorator(cls):
            return cls

        return decorator

    batch_size = 4
    seq_len = 10
    input_dim = 32
    hidden_dim = 64
    output_dim = 16
    num_layers = 2

    model = RNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
    )

    # forward pass
    x = torch.randn(batch_size, seq_len, input_dim)
    y = model(x)
    assert y.shape == (batch_size, seq_len, output_dim)

    # forward with initial hidden state
    h0 = torch.zeros(num_layers, batch_size, hidden_dim)
    y = model(x, h0)
    assert y.shape == (batch_size, seq_len, output_dim)

    # gradient flow
    x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    loss = model(x).sum()
    loss.backward()

    assert x.grad is not None
    for p in model.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()

    # variable batch & sequence sizes
    for bs, sl in [(1, 5), (8, 20)]:
        x = torch.randn(bs, sl, input_dim)
        y = model(x)
        assert y.shape == (bs, sl, output_dim)

    # numerical check vs PyTorch RNN (single-layer)
    torch.manual_seed(0)

    test_rnn = RNN(
        input_dim=16,
        hidden_dim=32,
        output_dim=32,
        num_layers=1,
    )

    ref_rnn = nn.RNN(
        input_size=16,
        hidden_size=32,
        batch_first=True,
        nonlinearity="tanh",
    )

    with torch.no_grad():
        test_rnn.w_ih[0].copy_(ref_rnn.weight_ih_l0)
        test_rnn.w_hh[0].copy_(ref_rnn.weight_hh_l0)
        test_rnn.b_ih[0].copy_(ref_rnn.bias_ih_l0)
        test_rnn.b_hh[0].copy_(ref_rnn.bias_hh_l0)
        test_rnn.h2o.weight.copy_(torch.eye(32))
        test_rnn.h2o.bias.zero_()

    x = torch.randn(2, 5, 16)
    h0 = torch.zeros(1, 2, 32)

    y_test = test_rnn(x, h0)
    y_ref, _ = ref_rnn(x, h0)

    assert torch.allclose(y_test, y_ref, atol=1e-5)

    if torch.cuda.is_available():
        model = model.cuda()
        x = torch.randn(batch_size, seq_len, input_dim, device="cuda")
        y = model(x)
        assert y.is_cuda

    print("All RNN tests passed.")

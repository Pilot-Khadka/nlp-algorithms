from typing import List, Tuple, Optional


import torch
import torch.nn as nn


from engine.registry import register_model
from engine.model_factory import BaseModel

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class LockedDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self._mask: Optional[torch.Tensor] = None
        self._cached_batch_size: Optional[int] = None
        self._cached_feat_size: Optional[int] = None

    def reset_mask(self):
        self._mask = None
        self._cached_batch_size = None
        self._cached_feat_size = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask_shape: tuple[int, ...]  # allow 2D or 3D

        if x.dim() == 3:
            batch_size, _, feat_size = x.size()
            mask_shape = (batch_size, 1, feat_size)  # broadcast across time
        else:
            batch_size, feat_size = x.size()
            mask_shape = (batch_size, feat_size)

        need_new_mask = (
            self._mask is None
            or self._cached_batch_size != batch_size
            or self._cached_feat_size != feat_size
            or self._mask.device != x.device
        )

        if need_new_mask:
            self._mask = x.new_empty(mask_shape).bernoulli_(1 - self.p) / (1 - self.p)
            self._cached_batch_size = batch_size
            self._cached_feat_size = feat_size

        return x * self._mask


class WordDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask = x.new_empty(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)
        return x * mask


@register_model("lstm")
class LSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        embed_dropout: float = 0.1,
        recurrent_dropout: float = 0.25,
        output_dropout: float = 0.5,
        tie_weights: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.tie_weights = tie_weights

        self.embedding = kwargs.get("embedding_layer", None)
        if self.embedding is None:
            self.embedding = nn.Embedding(output_dim, input_dim)
        assert self.embedding is not None
        self.embed_dropout = WordDropout(embed_dropout)

        # persist masks across BPTT until reset_dropout_masks() is called
        self.layer_dropouts = nn.ModuleList(
            [LockedDropout(dropout) for _ in range(num_layers)]
        )
        self.recurrent_dropouts = nn.ModuleList(
            [LockedDropout(recurrent_dropout) for _ in range(num_layers)]
        )
        self.output_dropout_module = LockedDropout(output_dropout)

        self._build_layers()
        self._setup_output_layers()
        self._initialize_weights()

    def _build_layers(self):
        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.bias = nn.ParameterList()

        for layer in range(self.num_layers):
            input_size = self.input_dim if layer == 0 else self.hidden_dim
            H = self.hidden_dim
            self.w_ih.append(nn.Parameter(torch.empty(input_size, 4 * H)))
            self.w_hh.append(nn.Parameter(torch.empty(H, 4 * H)))
            self.bias.append(nn.Parameter(torch.empty(4 * H)))

    def _setup_output_layers(self):
        assert self.embedding is not None
        if self.tie_weights:
            if self.hidden_dim == self.input_dim:
                self.proj = None
                self.fc = nn.Linear(self.hidden_dim, self.output_dim)
                self.fc.weight = self.embedding.weight  # Tie weights
                print(f"[Weight Tying] Direct: hidden_dim={self.hidden_dim}")
            else:
                self.proj = nn.Linear(self.hidden_dim, self.input_dim, bias=False)
                self.fc = nn.Linear(self.input_dim, self.output_dim)
                self.fc.weight = self.embedding.weight
                print(
                    f"[Weight Tying] With projection: {self.hidden_dim} -> {self.input_dim}"
                )
        else:
            self.proj = None
            self.fc = nn.Linear(self.hidden_dim, self.output_dim)
            print("[Weight Tying] Disabled")

    def _initialize_weights(self):
        init_range = 0.1

        assert self.embedding is not None
        nn.init.uniform_(self.embedding.weight, -init_range, init_range)

        H = self.hidden_dim
        for w_ih, w_hh, bias in zip(self.w_ih, self.w_hh, self.bias):
            nn.init.uniform_(w_ih, -init_range, init_range)
            nn.init.uniform_(w_hh, -init_range, init_range)
            nn.init.zeros_(bias)
            bias.data[H : 2 * H] = 1.0  # forget gate
            bias.data[3 * H : 4 * H] = 0.0  # output gate

        if self.proj is not None:
            nn.init.uniform_(self.proj.weight, -init_range, init_range)

        nn.init.zeros_(self.fc.bias)

    def init_hidden(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        hidden = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        cell = [
            torch.zeros(batch_size, self.hidden_dim, device=device, dtype=dtype)
            for _ in range(self.num_layers)
        ]
        return hidden, cell

    def detach_hidden(
        self,
        states: Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]],
    ) -> Optional[Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        if states is None:
            return None
        hidden, cell = states
        return ([h.detach() for h in hidden], [c.detach() for c in cell])

    def reset_dropout_masks(self):
        for module in self.layer_dropouts:
            module.reset_mask()
        for module in self.recurrent_dropouts:
            module.reset_mask()
        self.output_dropout_module.reset_mask()

    def _forward(
        self,
        x: torch.Tensor,
        hidden: List[torch.Tensor],
        cell: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        batch_size, seq_len, _ = x.size()
        H = self.hidden_dim
        L = self.num_layers

        h = [h_i.clone() for h_i in hidden]
        c = [c_i.clone() for c_i in cell]

        layer_input = x

        for layer_idx in range(L):
            w_ih = self.w_ih[layer_idx]
            w_hh = self.w_hh[layer_idx]
            bias = self.bias[layer_idx]
            recurrent_dropout = self.recurrent_dropouts[layer_idx]

            # Precompute input projection for entire sequence
            x_proj = layer_input @ w_ih  # (batch, seq, 4*H)

            outputs_t = torch.empty(
                batch_size, seq_len, H, device=x.device, dtype=x.dtype
            )

            for t in range(seq_len):
                # Recurrent dropout on h before h @ w_hh
                h_dropped = recurrent_dropout(h[layer_idx])

                gates = x_proj[:, t, :] + h_dropped @ w_hh + bias

                i = torch.sigmoid(gates[:, :H])
                f = torch.sigmoid(gates[:, H : 2 * H])
                g = torch.tanh(gates[:, 2 * H : 3 * H])
                o = torch.sigmoid(gates[:, 3 * H :])

                c[layer_idx] = f * c[layer_idx] + i * g
                h[layer_idx] = o * torch.tanh(c[layer_idx])

                outputs_t[:, t, :] = h[layer_idx]

            if layer_idx < L - 1:
                outputs_t = self.layer_dropouts[layer_idx](outputs_t)

            layer_input = outputs_t

        return outputs_t, h, c

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        cell: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[List[torch.Tensor], List[torch.Tensor]]]:
        batch_size, seq_len = x.shape

        assert self.embedding is not None
        emb = self.embedding(x)
        emb = self.embed_dropout(emb)

        if hidden is None or cell is None:
            hidden, cell = self.init_hidden(batch_size, x.device, emb.dtype)

        if hidden[0].size(0) != batch_size:
            hidden, cell = self.init_hidden(batch_size, x.device, emb.dtype)

        lstm_out, hidden, cell = self._forward(emb, hidden, cell)
        lstm_out = self.output_dropout_module(lstm_out)

        if self.proj is not None:
            lstm_out = self.proj(lstm_out)

        output = self.fc(lstm_out)
        return output, (hidden, cell)


class LSTMNaive(BaseModel):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        input_size = input_dim + hidden_dim
        self.embedding = kwargs.get("embedding_layer", None)
        self.forget_gate = nn.Linear(input_size, hidden_dim)
        self.input_gate1 = nn.Linear(input_size, hidden_dim)
        self.input_gate2 = nn.Linear(input_size, hidden_dim)
        self.output_gate = nn.Linear(input_size, hidden_dim)

        self.fc = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        else:
            h_t = hidden
        c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            combined = torch.cat([x_t, h_t], dim=1)

            f_t = self.sigmoid(self.forget_gate(combined))
            i_t = self.sigmoid(self.input_gate1(combined))
            g_t = self.tanh(self.input_gate2(combined))
            o_t = self.sigmoid(self.output_gate(combined))

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * self.tanh(c_t)
            outputs.append(h_t.unsqueeze(1))  # (B, 1, H)

        out = torch.cat(outputs, dim=1)
        return self.fc(out) if self.fc else out


if __name__ == "__main__":
    model = LSTM(input_dim=10, hidden_dim=20, output_dim=5)
    print(model)

    x_dummy = torch.randn(3, 7, 10)  # (batch_size, seq_len, input_dim)
    output = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())
    output.mean().backward()  # dummy loss

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

    x = torch.randn(100, 7, 10)
    y = (x.sum(dim=(1, 2)) > 0).long()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

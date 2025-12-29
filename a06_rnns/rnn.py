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
        use_compile=True,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = kwargs.get("embedding_layer", None)
        if self.embedding is not None:
            self.embedding.weight.requires_grad = False
            self.input_dim = self.embedding.embedding_dim
        else:
            self.input_dim = input_dim

        self.w_ih = nn.ParameterList()
        self.w_hh = nn.ParameterList()
        self.b_ih = nn.ParameterList()
        self.b_hh = nn.ParameterList()

        for layer in range(num_layers):
            input_size = self.input_dim if layer == 0 else hidden_dim
            self.w_ih.append(nn.Parameter(torch.empty(hidden_dim, input_size)))
            self.w_hh.append(nn.Parameter(torch.empty(hidden_dim, hidden_dim)))
            self.b_ih.append(nn.Parameter(torch.empty(hidden_dim)))
            self.b_hh.append(nn.Parameter(torch.empty(hidden_dim)))

        self.h2o = nn.Linear(hidden_dim, output_dim)
        self._reset_parameters()

        self._use_compile = use_compile
        self._compiled = False

    def _reset_parameters(self):
        stdv = 1.0 / (self.hidden_dim**0.5)
        for p in self.parameters():
            nn.init.uniform_(p, -stdv, stdv)

    def _rnn_forward_impl(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()

        # pre-allocate outputs
        outputs = torch.empty(
            batch_size, seq_len, self.hidden_dim, device=x.device, dtype=x.dtype
        )

        # unbind hidden states
        h_layers = list(hidden.unbind(0))

        for t in range(seq_len):
            x_t = x[:, t, :]

            for layer in range(self.num_layers):
                h_new = torch.tanh(
                    torch.addmm(self.b_ih[layer], x_t, self.w_ih[layer].t())
                    + torch.addmm(
                        self.b_hh[layer], h_layers[layer], self.w_hh[layer].t()
                    )
                )
                h_layers[layer] = h_new
                x_t = h_new

            outputs[:, t, :] = x_t

        return outputs

    def _compile(self):
        if self._use_compile and not self._compiled:
            self._rnn_forward_impl = torch.compile(
                self._rnn_forward_impl,
                mode="reduce-overhead",
                fullgraph=False,
            )
            self._compiled = True

    def forward(self, x, hidden=None):
        if self.embedding:
            x = self.embedding(x)
        else:
            if x.dim() == 2:
                x = x.unsqueeze(-1)
            x = x.float()

        assert x.size(-1) == self.input_dim

        batch_size, seq_len, _ = x.size()

        if hidden is None:
            hidden = torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_dim,
                device=x.device,
                dtype=x.dtype,
            )

        self._compile()

        rnn_out = self._rnn_forward_impl(x, hidden)
        outputs = self.h2o(rnn_out.view(-1, self.hidden_dim))
        return outputs.view(batch_size, seq_len, -1)


if __name__ == "__main__":
    # remove registry dependency for standalone testing
    class BaseModel(nn.Module):
        pass

    def register_model(name, *flag):
        def decorator(cls):
            return cls

        return decorator

    print("=" * 60)
    print("RNN Model Test Suite")
    print("=" * 60)

    batch_size = 4
    seq_len = 10
    input_dim = 32
    hidden_dim = 64
    output_dim = 16
    num_layers = 2

    print("\n[Test 1] Model Instantiation...")
    model = RNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        use_compile=False,
    )
    print(
        f"  Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    print("\n[Test 2] Basic Forward Pass...")
    x = torch.randn(batch_size, seq_len, input_dim)
    output = model(x)
    expected_shape = (batch_size, seq_len, output_dim)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )
    print(f"  Input shape:  {tuple(x.shape)}")
    print(f"  Output shape: {tuple(output.shape)}")

    print("\n[Test 3] Forward Pass with Initial Hidden State...")
    hidden = torch.zeros(num_layers, batch_size, hidden_dim)
    output_with_hidden = model(x, hidden)
    assert output_with_hidden.shape == expected_shape
    print(f"  Hidden shape: {tuple(hidden.shape)}")
    print(f"  Output shape: {tuple(output_with_hidden.shape)}")

    print("\n[Test 4] Gradient Flow (Backpropagation)...")
    model.zero_grad()
    x_grad = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
    output = model(x_grad)
    loss = output.sum()
    loss.backward()

    assert x_grad.grad is not None, "Input gradient is None"
    assert not torch.isnan(x_grad.grad).any(), "NaN in input gradients"

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient is None for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN in gradient for {name}"
    print("   All gradients computed successfully")
    print("   No NaN values in gradients")

    print("\n[Test 5] Variable Batch Sizes...")
    for bs in [1, 8, 16, 32]:
        x_var = torch.randn(bs, seq_len, input_dim)
        out_var = model(x_var)
        assert out_var.shape == (bs, seq_len, output_dim)
        print(f"  Batch size {bs:2d}: output shape {tuple(out_var.shape)}")

    print("\n[Test 6] Variable Sequence Lengths...")
    for sl in [1, 5, 20, 50]:
        x_var = torch.randn(batch_size, sl, input_dim)
        out_var = model(x_var)
        assert out_var.shape == (batch_size, sl, output_dim)
        print(f"   Seq length {sl:2d}: output shape {tuple(out_var.shape)}")

    print("\n[Test 7] Numerical Comparison with PyTorch RNN...")
    torch.manual_seed(42)

    test_input_dim = 16
    test_hidden_dim = 32

    our_rnn = RNN(
        input_dim=test_input_dim,
        hidden_dim=test_hidden_dim,
        output_dim=test_hidden_dim,
        num_layers=1,
        use_compile=False,
    )

    torch_rnn = nn.RNN(
        input_size=test_input_dim,
        hidden_size=test_hidden_dim,
        num_layers=1,
        batch_first=True,
        nonlinearity="tanh",
    )

    with torch.no_grad():
        our_rnn.w_ih[0].copy_(torch_rnn.weight_ih_l0)
        our_rnn.w_hh[0].copy_(torch_rnn.weight_hh_l0)
        our_rnn.b_ih[0].copy_(torch_rnn.bias_ih_l0)
        our_rnn.b_hh[0].copy_(torch_rnn.bias_hh_l0)
        our_rnn.h2o.weight.copy_(torch.eye(test_hidden_dim))
        our_rnn.h2o.bias.zero_()

    test_x = torch.randn(2, 5, test_input_dim)
    test_h0 = torch.zeros(1, 2, test_hidden_dim)

    our_output = our_rnn(test_x, test_h0)
    torch_output, _ = torch_rnn(test_x, test_h0)

    max_diff = (our_output - torch_output).abs().max().item()
    print(f"   Max absolute difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Outputs differ too much: {max_diff}"
    print("  Outputs match PyTorch RNN!")

    print("\n[Test 8] Multi-layer RNN...")
    for n_layers in [1, 2, 3, 4]:
        multi_layer_rnn = RNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=n_layers,
            use_compile=False,
        )
        x_test = torch.randn(batch_size, seq_len, input_dim)
        out_test = multi_layer_rnn(x_test)
        assert out_test.shape == (batch_size, seq_len, output_dim)
        print(f"   {n_layers} layer(s): output shape {tuple(out_test.shape)}")

    print("\n[Test 9] GPU Test...")
    if torch.cuda.is_available():
        model_cuda = model.cuda()
        x_cuda = torch.randn(batch_size, seq_len, input_dim, device="cuda")
        output_cuda = model_cuda(x_cuda)

        assert output_cuda.device.type == "cuda"
        assert output_cuda.shape == expected_shape
        assert not torch.isnan(output_cuda).any()
        print("   GPU forward pass successful")
        print(f"   Output device: {output_cuda.device}")
    else:
        print("  CUDA not available, skipping GPU test")

    print("\n[Test 11] Different Data Types...")
    for dtype in [torch.float32, torch.float64]:
        model_dtype = RNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            use_compile=False,
        ).to(dtype)
        x_dtype = torch.randn(batch_size, seq_len, input_dim, dtype=dtype)
        out_dtype = model_dtype(x_dtype)
        assert out_dtype.dtype == dtype
        print(f"   {dtype}: output dtype matches")

    print(" All tests passed successfully!")

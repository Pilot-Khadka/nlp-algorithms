import torch
import torch.nn as nn

from nlp_algorithms.engine.registry import register_model


@register_model("gru", flags=["pytorch"])
class GRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, x, hidden=None):
        return self.gru(x, hidden)


@register_model("gru", flags=["pytorch", "bidirectional"])
class BiGRU(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = True

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x, hidden=None):
        return self.gru(x, hidden)


if __name__ == "__main__":
    model = GRU(input_dim=10, hidden_dim=20, num_layers=1)
    print(model)

    x_dummy = torch.randn(3, 7, 10)
    output, h = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())

    output.mean().backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

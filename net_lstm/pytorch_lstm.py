import torch
import torch.nn as nn
from engine.registry import register_model

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


@register_model("lstm", flags=["unidirectional", "pytorch"])
class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_layers,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = kwargs.get("embedding_layer", None)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden=None):
        if self.embedding is not None:
            x = self.embedding(x)

        output_seq, (hn, cn) = self.lstm(x, hidden)

        output_seq = self.fc(output_seq)

        return output_seq


if __name__ == "__main__":
    model = LSTM(input_dim=10, hidden_dim=20, output_dim=5, num_layers=1)
    print(model)

    x_dummy = torch.randn(3, 7, 10)
    output = model(x_dummy)
    print("Output shape:", output.shape)
    print("Any NaNs in output?", torch.isnan(output).any().item())
    print("Output max abs:", output.abs().max().item())

    output.mean().backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"{name} grad norm:", param.grad.norm().item())
        else:
            print(f"{name} has no grad")

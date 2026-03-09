import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, shape, eps=1e-5):
        super().__init__()
        self.eps = eps  # for non-zero div.
        self.gamma = nn.Parameter(torch.ones(shape))  # scale
        self.beta = nn.Parameter(torch.zeros(shape))  # shift

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)

        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_normalized + self.beta


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 4)
    print("Input:\n", x)

    ln = LayerNorm(shape=4)

    output = ln(x)
    print("Output:\n", output)

    with torch.no_grad():
        normalized = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(
            x.var(dim=-1, unbiased=False, keepdim=True) + ln.eps
        )
        print("Manually Normalized :\n", normalized)

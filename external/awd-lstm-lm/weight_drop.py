import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightDrop(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        weights: list[str],
        dropout: float = 0.0,
        variational: bool = False,
    ):
        super().__init__()
        self.module = module
        self.weights = weights
        self.dropout = dropout
        self.variational = variational
        self._setup()

    def _setup(self):
        if isinstance(self.module, nn.RNNBase):
            self.module.flatten_parameters = lambda: None

        for name in self.weights:
            w = getattr(self.module, name)
            del self.module._parameters[name]
            self.module.register_parameter(name + "_raw", nn.Parameter(w.data))

    def _drop_weight(self, weight: Tensor) -> Tensor:
        if self.variational:
            mask = weight.new_ones(weight.size(0), 1)
            mask = F.dropout(mask, p=self.dropout, training=self.training)
            return mask.expand_as(weight) * weight
        return F.dropout(weight, p=self.dropout, training=self.training)

    def _apply_weights(self):
        for name in self.weights:
            raw_w = getattr(self.module, name + "_raw")
            self.module.__dict__[name] = self._drop_weight(raw_w)

    def forward(self, *args, **kwargs):
        self._apply_weights()
        return self.module(*args, **kwargs)


if __name__ == "__main__":
    import torch
    from weight_drop import WeightDrop

    # Input is (seq, batch, input)
    x = torch.autograd.Variable(torch.randn(2, 1, 10)).cuda()
    h0 = None

    ###

    print("Testing WeightDrop")
    print("=-=-=-=-=-=-=-=-=-=")

    ###

    print("Testing WeightDrop with Linear")

    lin = WeightDrop(torch.nn.Linear(10, 10), ["weight"], dropout=0.9)
    lin.cuda()
    run1 = [x.sum() for x in lin(x).data]
    run2 = [x.sum() for x in lin(x).data]

    print("All items should be different")
    print("Run 1:", run1)
    print("Run 2:", run2)

    assert run1[0] != run2[0]
    assert run1[1] != run2[1]

    print("---")

    ###

    print("Testing WeightDrop with LSTM")

    wdrnn = WeightDrop(torch.nn.LSTM(10, 10), ["weight_hh_l0"], dropout=0.9)
    wdrnn.cuda()

    run1 = [x.sum() for x in wdrnn(x, h0)[0].data]
    run2 = [x.sum() for x in wdrnn(x, h0)[0].data]

    print("First timesteps should be equal, all others should differ")
    print("Run 1:", run1)
    print("Run 2:", run2)

    # First time step, not influenced by hidden to hidden weights, should be equal
    assert run1[0] == run2[0]
    # Second step should not
    assert run1[1] != run2[1]

    print("---")

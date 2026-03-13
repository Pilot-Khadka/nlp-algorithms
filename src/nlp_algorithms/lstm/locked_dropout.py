from typing import Tuple, Optional, Union


import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    """Dropout with mask fixed across timesteps within a single forward pass."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p
        self._mask: Optional[torch.Tensor] = None

    def reset_mask(self) -> None:
        self._mask = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask_shape: Union[Tuple[int, int, int], Tuple[int, int]]
        if x.dim() == 3:
            batch_size, _, feat_size = x.size()
            mask_shape = (batch_size, 1, feat_size)
        else:
            batch_size, feat_size = x.size()
            mask_shape = (batch_size, feat_size)

        if self._mask is None:
            self._mask = x.new_empty(mask_shape).bernoulli_(1 - self.p) / (1 - self.p)

        return x * self._mask

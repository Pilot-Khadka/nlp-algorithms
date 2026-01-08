from typing import Any, Dict, Tuple, List, Optional


from abc import ABC, abstractmethod


import torch
from torch import nn


class BaseTask(ABC):
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def get_output_dim(self, dataset_bundle: Any) -> int:
        raise NotImplementedError()

    @abstractmethod
    def train_step(
        self,
        batch: Any,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip: Optional[float],
        device: torch.device,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_step(
        self,
        batch: Any,
        model: nn.Module,
        device: torch.device,
        metrics_list: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        pass

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        metrics_list: List[str],
    ) -> Dict[str, float]:
        return {}

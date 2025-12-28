from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List


import torch
from torch import nn


class BaseTask(ABC):
    @abstractmethod
    def name(self) -> str:
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
        device: torch.device,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def eval_step(
        self,
        batch: Any,
        model: nn.Module,
        device: torch.device,
        metrics: List[str],
    ) -> Tuple[float, Dict[str, float]]:
        pass

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, float]:
        return {}

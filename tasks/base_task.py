from typing import Any, Dict, List


from abc import ABC, abstractmethod


import torch
from torch import nn


class BaseTask(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_output_dim(self, dataset_bundle: Any) -> int: ...

    @abstractmethod
    def train_one_epoch(
        self,
        model,
        train_loader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        total_epochs: int,
        grad_clip: float,
        use_ddp: bool = False,
        is_main: bool = True,
    ) -> float: ...

    @abstractmethod
    def evaluate_one_epoch(
        self,
        model,
        valid_loader,
        device: torch.device,
        epoch: int,
        total_epochs: int,
        metrics_tracker,
        use_ddp: bool = False,
        is_main: bool = True,
    ) -> tuple[float, Dict[str, float]]: ...

    def compute_metrics(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        metrics_list: List[str],
    ) -> Dict[str, float]:
        return {}

    def on_epoch_start(self, model: nn.Module, training: bool = True) -> None:
        pass

    def on_epoch_end(self, model: nn.Module, training: bool = True) -> None:
        pass

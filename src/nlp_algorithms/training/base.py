"""
Note: not shuffling causes always returning the first prefetched batch, repeating it.
This can be solved by shuffling the train dataset but need to check for language modeling task
"""

from abc import ABC, abstractmethod

import torch
import torch.distributed as dist

from nlp_algorithms.util.metric import MetricsTracker
from nlp_algorithms.util.logger import setup_logging


class BaseTrainer(ABC):
    def __init__(
        self,
        config,
        builder,
        gpu_id: int = 0,
        use_ddp: bool = False,
    ):
        self.config = config
        self.logger = setup_logging()
        self.device = torch.device(  # type: ignore[assignment]
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        self.gpu_id = gpu_id
        self.use_ddp = use_ddp
        self.is_main = not use_ddp or gpu_id == 0

        self.model = builder.model
        self.optimizer = builder.optimizer
        self.scheduler = builder.scheduler
        self.train_loader = builder.train_loader
        self.test_loader = builder.test_loader
        self.task = builder.task
        self.criterion = builder.criterion

        self.metrics_tracker = MetricsTracker(builder.metric_names)

        self.grad_clip = self.config.train.grad_clip
        self.best_valid_loss = float("inf")

    def train(self) -> None:
        """Main training loop."""
        for epoch in range(self.config.train.epochs):
            avg_train_loss = self.train_one_epoch(epoch, self.config.train.epochs)

            avg_valid_loss, avg_metrics = self.evaluate_one_epoch(
                epoch, self.config.train.epochs
            )

            if self.scheduler is not None:
                self.scheduler.step(avg_valid_loss)

            if self.is_main:
                self._log_epoch_results(
                    epoch, avg_train_loss, avg_valid_loss, avg_metrics
                )

                if avg_valid_loss < self.best_valid_loss:
                    self.best_valid_loss = avg_valid_loss

            if self.use_ddp:
                dist.barrier()

    def _tqdm_format_metrics(self, metrics: dict, max_items: int = 3) -> str:
        if not metrics:
            return ""

        items = list(metrics.items())[:max_items]

        parts = [f"{k}={v:.4f}" for k, v in items]
        result = " | ".join(parts)

        if len(result) > 45:
            result = result[:42] + "..."

        return result

    def _log_epoch_results(
        self, epoch: int, train_loss: float, valid_loss: float, metrics: dict
    ) -> None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Valid Loss: {valid_loss:.4f}, "
            f"{metrics_str}"
        )

        print("\n" + "-" * 70)
        print(f"Epoch {epoch + 1} [Validation] Metrics")
        print("-" * 70)
        print(self.metrics_tracker.format_metrics(metrics))
        print("-" * 70)

    @abstractmethod
    def train_one_epoch(self, epoch: int, total_epochs: int) -> float:
        pass

    @abstractmethod
    def evaluate_one_epoch(self, epoch: int, total_epochs: int) -> tuple[float, dict]:
        pass

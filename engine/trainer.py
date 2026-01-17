from typing import Dict, Any, Optional, Union

from pathlib import Path

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from task.base_task import BaseTask
from util.metric import MetricsTracker
from util.util import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(
        self,
        model,
        task: BaseTask,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        config,
        metrics: Dict[str, Any],
        logger: Any,
        gpu_id: int = 0,
        use_ddp: bool = False,
        resume_from: Optional[Union[str, Path]] = None,
    ) -> None:
        self.gpu_id = gpu_id
        # pyrefly: ignore [read-only]
        self.device = torch.device(f"cuda:{gpu_id}")
        self.use_ddp = use_ddp
        self.is_main = not use_ddp or gpu_id == 0

        self.model = model.to(self.device)
        if use_ddp:
            self.model = DDP(self.model, device_ids=[gpu_id])

        self.task = task
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.config = config
        self.logger = logger

        if (
            hasattr(config, "optimizer")
            and config.optimizer.lower() == "sgd"
            and hasattr(config, "lr_decay")
            and config.lr_decay
        ):
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=config.lr_decay,
                patience=1,
            )
        else:
            # pyrefly: ignore [bad-assignment]
            self.scheduler = None

        self.start_epoch = 0
        self.best_valid_loss = float("inf")

        if resume_from is not None:
            self.logger.info(f"Loading model from checkpoint:{resume_from}")
            start_epoch, best_valid_loss = load_checkpoint(
                resume_from,
                model=self.model.module if isinstance(self.model, DDP) else self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                device=self.device,
            )
            self.start_epoch = start_epoch
            self.best_valid_loss = best_valid_loss

        metric_names = list(metrics.keys()) if isinstance(metrics, dict) else metrics
        self.metrics_tracker = MetricsTracker(metric_names)

        if self.is_main:
            self.checkpoint_dir = Path(config.get("checkpoints", "./checkpoints"))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_checkpoint(
        self,
        epoch: int,
        avg_valid_loss: float,
        avg_train_loss: float,
        avg_metrics: Dict[str, float],
        checkpoint_name: str = "last.pt",
    ):
        if not self.is_main:
            return

        unwrapped_model = (
            self.model.module if isinstance(self.model, DDP) else self.model
        )

        additional_info = {
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            "metrics": avg_metrics,
        }

        save_checkpoint(
            checkpoint_path=self.checkpoint_dir / checkpoint_name,
            epoch=epoch + 1,
            optimizer=self.optimizer,
            model=unwrapped_model,
            scheduler=self.scheduler,
            best_valid_loss=self.best_valid_loss,
            additional_info=additional_info,
        )

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.config["epochs"]):
            avg_train_loss = self.task.train_one_epoch(
                model=self.model,
                train_loader=self.train_loader,
                optimizer=self.optimizer,
                device=self.device,
                epoch=epoch,
                total_epochs=self.config["epochs"],
                grad_clip=self.config.grad_clip,
                use_ddp=self.use_ddp,
                is_main=self.is_main,
            )

            avg_valid_loss, avg_metrics = self.task.evaluate_one_epoch(
                model=self.model,
                valid_loader=self.valid_loader,
                device=self.device,
                epoch=epoch,
                total_epochs=self.config["epochs"],
                metrics_tracker=self.metrics_tracker,
                use_ddp=self.use_ddp,
                is_main=self.is_main,
            )

            if self.scheduler is not None:
                self.scheduler.step(avg_valid_loss)

            if self.is_main:
                metrics_str = ", ".join(
                    [f"{k}: {v:.4f}" for k, v in avg_metrics.items()]
                )
                self.logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Valid Loss: {avg_valid_loss:.4f}, "
                    f"{metrics_str}"
                )
                self.metrics_tracker.print_summary(epoch + 1, "Validation")

                self._save_checkpoint(
                    epoch, avg_valid_loss, avg_train_loss, avg_metrics, "last.pt"
                )

                if avg_valid_loss < self.best_valid_loss:
                    self.best_valid_loss = avg_valid_loss
                    self._save_checkpoint(
                        epoch, avg_valid_loss, avg_train_loss, avg_metrics, "best.pt"
                    )
                    # self.logger.info(
                    #     f"Saved new best model with valid loss: {self.best_valid_loss:.4f}"
                    # )

            if self.use_ddp:
                dist.barrier()

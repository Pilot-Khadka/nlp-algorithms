from typing import Dict, Any, Optional, Union


from pathlib import Path
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tasks.base_task import BaseTask
from utils.metrics import MetricsTracker
from utils.utils import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(
        self,
        model: nn.Module,
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
            self.scheduler = None

        self.start_epoch = 0
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
        self.best_valid_loss = float("inf")

        if self.is_main:
            self.checkpoint_dir = Path(config.get("checkpoints", "./checkpoints"))
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _run_train_epoch(self, epoch: int) -> float:
        self.model.train()

        if self.use_ddp:
            self.train_loader.sampler.set_epoch(epoch)

        total_train_loss = 0

        if self.is_main:
            train_progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{self.config['epochs']} [Train]",
                leave=False,
            )
        else:
            train_progress = self.train_loader

        for batch_idx, batch in enumerate(train_progress):
            loss = self.task.train_step(batch, self.model, self.optimizer, self.device)
            total_train_loss += loss

            if self.is_main and hasattr(train_progress, "set_postfix"):
                train_progress.set_postfix(
                    {
                        "Loss": f"{loss:.4f}",
                        "Avg Loss": f"{total_train_loss / (batch_idx + 1):.4f}",
                    }
                )

        avg_train_loss = total_train_loss / len(self.train_loader)

        if self.use_ddp:
            avg_train_loss_tensor = torch.tensor(avg_train_loss, device=self.device)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = avg_train_loss_tensor.item()

        return avg_train_loss

    def _run_valid_epoch(self, epoch: int) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        total_valid_loss = 0.0
        self.metrics_tracker.reset()

        if self.is_main:
            valid_progress = tqdm(
                self.valid_loader,
                desc=f"Epoch {epoch + 1}/{self.config['epochs']} [Valid]",
                leave=False,
            )
        else:
            valid_progress = self.valid_loader

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_progress):
                loss, batch_metrics = self.task.eval_step(
                    batch, self.model, self.device, self.metrics_tracker.metric_names
                )
                total_valid_loss += loss
                batch_size = batch[0].size(0)
                self.metrics_tracker.update(batch_metrics, batch_size)

                if self.is_main and hasattr(valid_progress, "set_postfix"):
                    current_metrics = self.metrics_tracker.get_averages()
                    postfix = {
                        "Val Loss": f"{loss:.4f}",
                        "Avg Val Loss": f"{total_valid_loss / (batch_idx + 1):.4f}",
                    }
                    postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                    valid_progress.set_postfix(postfix)

        avg_valid_loss = total_valid_loss / len(self.valid_loader)

        if self.use_ddp:
            avg_valid_loss_tensor = torch.tensor(avg_valid_loss, device=self.device)
            dist.all_reduce(avg_valid_loss_tensor, op=dist.ReduceOp.AVG)
            avg_valid_loss = avg_valid_loss_tensor.item()

            avg_metrics = self.metrics_tracker.get_averages()
            for key in avg_metrics:
                metric_tensor = torch.tensor(avg_metrics[key], device=self.device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
                avg_metrics[key] = metric_tensor.item()
        else:
            avg_metrics = self.metrics_tracker.get_averages()

        return avg_valid_loss, avg_metrics

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
            epoch=epoch + 1,  # next epoch to run
            optimizer=self.optimizer,
            model=unwrapped_model,
            scheduler=self.scheduler,
            best_valid_loss=self.best_valid_loss,
            additional_info=additional_info,
        )

    def train(self) -> None:
        for epoch in range(self.start_epoch, self.config["epochs"]):
            avg_train_loss = self._run_train_epoch(epoch)
            avg_valid_loss, avg_metrics = self._run_valid_epoch(epoch)

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
                    self.logger.info(
                        f"Saved new best model with valid loss: {self.best_valid_loss:.4f}"
                    )

            if self.use_ddp:
                dist.barrier()

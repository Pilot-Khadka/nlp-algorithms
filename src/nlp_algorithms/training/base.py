"""
Note: not shuffling causes always returning the first prefetched batch, repeating it.
This can be solved by shuffling the train dataset but need to check for language modeling task
"""

from abc import ABC, abstractmethod

import os
import time

import torch
import torch.distributed as dist

from nlp_algorithms.util.logger import setup_logging, fmt_time
from nlp_algorithms.util.general_util import resolve_checkpoint_dir, unwrap_model


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
        self.metrics = builder.metric_names
        self.task = builder.task
        self.criterion = builder.criterion
        self.grad_clip = self.config.train.grad_clip
        self.best_valid_loss = float("inf")
        self._avg_train_time = 0.0
        self._avg_val_time = 0.0
        self.checkpoint_dir = resolve_checkpoint_dir(config)

        # train.amp: bf16 where the gpu has it, t4/p100 use fp16 with a grad scaler
        self.amp_dtype = None
        if config.train.get("amp", False) and self.device.type == "cuda":
            if torch.cuda.is_bf16_supported(including_emulation=False):
                self.amp_dtype = torch.bfloat16
            else:
                self.amp_dtype = torch.float16
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_dtype == torch.float16
        )
        # a log line every log_every train steps when there is no progress bar
        self.log_every = config.train.get("log_every", None)

        if self.is_main:
            world_size = dist.get_world_size() if use_ddp else 1
            amp = {torch.bfloat16: "bf16", torch.float16: "fp16"}.get(
                self.amp_dtype, "off"
            )
            self.logger.info(
                f"train {config.task.name}: {config.train.epochs} epochs x "
                f"{len(self.train_loader)} steps, batch {config.train.batch_size} x "
                f"{world_size} {self.device.type}, amp {amp}, "
                f"{getattr(self.train_loader, 'num_workers', 0)} loader workers each, "
                f"checkpoints in {self.checkpoint_dir}"
            )

    def warmup_linear_decay(self):
        # bert schedule: linear warmup then linear decay to 0, stepped every batch
        warmup = self.config.train.get("warmup", 0)
        total = max(1, self.config.train.epochs * len(self.train_loader))

        def lr_lambda(step):
            if step < warmup:
                return (step + 1) / warmup
            return max(0.0, (total - step) / max(1, total - warmup))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype or torch.bfloat16,
            enabled=self.amp_dtype is not None,
        )

    def backward_step(self, loss) -> None:
        # backward, clip and step, the scaler is a no-op without fp16
        self.scaler.scale(loss).backward()
        if self.grad_clip:
            # clip the real gradients, not the scaled ones
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def train(self) -> None:
        for epoch in range(self.config.train.epochs):
            avg_train_loss, train_time = self.train_one_epoch(
                epoch, self.config.train.epochs
            )
            avg_valid_loss, val_time, avg_metrics = self.evaluate_one_epoch(
                epoch, self.config.train.epochs
            )

            n = epoch + 1
            self._avg_train_time += (train_time - self._avg_train_time) / n
            self._avg_val_time += (val_time - self._avg_val_time) / n

            if self.scheduler is not None:
                self.scheduler.step(avg_valid_loss)

            if self.is_main:
                is_best = avg_valid_loss < self.best_valid_loss
                self._log_epoch_results(
                    epoch,
                    avg_train_loss,
                    train_time,
                    val_time,
                    avg_valid_loss,
                    avg_metrics,
                    is_best,
                )
                if is_best:
                    self.best_valid_loss = avg_valid_loss
                    self.save_model("model_best.pt")

            if self.use_ddp:
                dist.barrier()

        if self.is_main:
            self.save_model("model_final.pt")

    def save_model(self, file_name: str) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, file_name)
        torch.save(unwrap_model(self.model).state_dict(), path)

    def _tqdm_format_metrics(self, metrics: dict, max_items: int = 3) -> str:
        if not metrics:
            return ""
        items = list(metrics.items())[:max_items]
        parts = [f"{k}={v:.4f}" for k, v in items]
        result = " | ".join(parts)
        if len(result) > 45:
            result = result[:42] + "..."
        return result

    def should_log_step(self, step: int, num_steps: int) -> bool:
        # step counts from 1, the last step is covered by the epoch line
        if not self.is_main or self.config.show_progress:
            return False
        log_every = self.log_every or max(1, num_steps // 10)
        return step % log_every == 0 and step < num_steps

    def log_step(
        self, epoch: int, step: int, num_steps: int, start_time: float, values: dict
    ) -> None:
        elapsed = time.perf_counter() - start_time
        values_str = " | ".join(f"{k} {v:.4f}" for k, v in values.items())
        lr = self.optimizer.param_groups[0]["lr"]
        self.logger.info(
            f"epoch {epoch + 1} step {step}/{num_steps} | {values_str} | "
            f"lr {lr:.2e} | {step / elapsed:.2f} steps/s | "
            f"epoch eta {fmt_time(elapsed / step * (num_steps - step))}"
        )

    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        train_time: float,
        val_time: float,
        valid_loss: float,
        metrics: dict,
        is_best: bool = False,
    ) -> None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        lr = self.optimizer.param_groups[0]["lr"]
        epochs_left = self.config.train.epochs - epoch - 1
        eta = epochs_left * (self._avg_train_time + self._avg_val_time)
        self.logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f} | "
            f"Train time: {train_time:.2f}s (avg: {self._avg_train_time:.2f}s) | "
            f"Valid Loss: {valid_loss:.4f} | "
            f"Valid time: {val_time:.2f}s (avg: {self._avg_val_time:.2f}s) | "
            f"{metrics_str} | lr {lr:.2e} | eta {fmt_time(eta)}"
            + (" | best, saved" if is_best else "")
        )

    @abstractmethod
    def train_one_epoch(self, epoch: int, total_epochs: int) -> tuple[float, float]:
        pass

    @abstractmethod
    def evaluate_one_epoch(
        self, epoch: int, total_epochs: int
    ) -> tuple[float, float, dict]:
        pass

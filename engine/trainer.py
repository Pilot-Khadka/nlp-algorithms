from typing import Dict, Any


import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from tasks.base_task import BaseTask
from utils.metrics import MetricsTracker
from utils.utils import save_checkpoint


def train(
    model: nn.Module,
    task: BaseTask,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: Any,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    metric_names = list(metrics.keys()) if isinstance(metrics, dict) else metrics
    metrics_tracker = MetricsTracker(metric_names)

    best_valid_loss = float("inf")
    checkpoint_dir = Path(config.get("checkpoints", "./checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        total_train_loss = 0
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]",
            leave=False,
        )
        for batch_idx, batch in enumerate(train_progress):
            loss = task.train_step(batch, model, optimizer, device)
            total_train_loss += loss
            train_progress.set_postfix(
                {
                    "Loss": f"{loss:.4f}",
                    "Avg Loss": f"{total_train_loss / (batch_idx + 1):.4f}",
                }
            )
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_valid_loss = 0.0
        metrics_tracker.reset()
        valid_progress = tqdm(
            valid_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']} [Valid]",
            leave=False,
        )
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_progress):
                loss, batch_metrics = task.eval_step(batch, model, device, metrics)
                total_valid_loss += loss
                batch_size = batch[0].size(0)
                metrics_tracker.update(batch_metrics, batch_size)
                current_metrics = metrics_tracker.get_averages()
                postfix = {
                    "Val Loss": f"{loss:.4f}",
                    "Avg Val Loss": f"{total_valid_loss / (batch_idx + 1):.4f}",
                }
                postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                valid_progress.set_postfix(postfix)

        avg_valid_loss = total_valid_loss / len(valid_loader)
        avg_metrics = metrics_tracker.get_averages()
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in avg_metrics.items()])
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Valid Loss: {avg_valid_loss:.4f}, "
            f"{metrics_str}"
        )
        metrics_tracker.print_summary(epoch + 1, "Validation")

        current_lr = optimizer.param_groups[0]["lr"]
        additional_info = {
            "train_loss": avg_train_loss,
            "valid_loss": avg_valid_loss,
            **avg_metrics,
        }

        save_checkpoint(
            checkpoint_path=checkpoint_dir / "last.pt",
            epoch=epoch + 1,
            lr=current_lr,
            optimizer=optimizer,
            model=model,
            metric_value=avg_valid_loss,
            additional_info=additional_info,
        )

        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            save_checkpoint(
                checkpoint_path=checkpoint_dir / "best.pt",
                epoch=epoch + 1,
                lr=current_lr,
                optimizer=optimizer,
                model=model,
                metric_value=avg_valid_loss,
                additional_info=additional_info,
            )
            logger.info(f"Saved new best model with valid loss: {best_valid_loss:.4f}")

from typing import Dict, Any, Union, cast, Optional, Tuple, List
from torch import Tensor


import inspect
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

from tasks.base_task import BaseTask
import utils.metrics as lm_metrics


class LanguageModelingTask(BaseTask):
    @property
    def name(self):
        return "language_modeling"

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.vocab_size

    def get_loss_fn(self, pad_idx=0):
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

    def on_epoch_start(self, model, training: bool = True) -> None:
        """
        Called at the start of each epoch.
        Resets model state for a fresh start on the data.
        """
        if hasattr(model, "reset_state"):
            # pyrefly: ignore [not-callable]
            model.reset_state()

    def on_epoch_end(self, model, training: bool = True) -> None:
        """Called at the end of each epoch."""
        pass

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
    ) -> float:
        model.train()
        criterion = self.get_loss_fn()

        if use_ddp:
            # pyrefly: ignore [missing-attribute]
            train_loader.sampler.set_epoch(epoch)

        self.on_epoch_start(model, training=True)

        total_train_loss = 0.0

        train_progress: Union[Any, tqdm]
        if is_main:
            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Train]",
                leave=False,
            )
        else:
            train_progress = train_loader

        hidden: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None
        cell: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None
        forward_params = inspect.signature(model.forward).parameters
        accepts_hidden = "hidden" in forward_params
        accepts_cell = "cell" in forward_params
        for batch_idx, batch in enumerate(train_progress):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            forward_kwargs = {}
            if accepts_hidden and hidden is not None:
                forward_kwargs["hidden"] = hidden
            if accepts_cell and cell is not None:
                forward_kwargs["cell"] = cell

            outputs = model(inputs, **forward_kwargs)

            # unpack outputs safely
            if isinstance(outputs, tuple):
                logits, state = outputs

                # LSTM: (hidden, cell)
                if isinstance(state, tuple):
                    hidden, cell = state
                # RNN / GRU: hidden only
                else:
                    hidden = state
                    cell = None
            else:
                logits = outputs
                hidden, cell = None, None

            # detach state between batches
            if hidden is not None:
                hidden = (
                    [h.detach() for h in hidden]
                    if isinstance(hidden, list)
                    else hidden.detach()
                )
            if cell is not None:
                cell = (
                    [c.detach() for c in cell]
                    if isinstance(cell, list)
                    else cell.detach()
                )

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            loss_value = loss.item()
            total_train_loss += loss_value

            if is_main and hasattr(train_progress, "set_postfix"):
                cast(tqdm, train_progress).set_postfix(
                    {
                        "Loss": f"{loss_value:.4f}",
                        "Avg Loss": f"{total_train_loss / (batch_idx + 1):.4f}",
                    }
                )

        self.on_epoch_end(model, training=True)

        avg_train_loss = total_train_loss / len(train_loader)

        if use_ddp:
            avg_train_loss_tensor = torch.tensor(avg_train_loss, device=device)
            dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
            avg_train_loss = avg_train_loss_tensor.item()

        return avg_train_loss

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
    ) -> tuple[float, Dict[str, float]]:
        """Evaluate for one complete epoch."""
        model.eval()
        criterion = self.get_loss_fn()
        total_valid_loss = 0.0
        metrics_tracker.reset()

        self.on_epoch_start(model, training=False)

        valid_progress: Union[Any, tqdm]
        if is_main:
            valid_progress = tqdm(
                valid_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Valid]",
                leave=False,
            )
        else:
            valid_progress = valid_loader

        hidden: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None
        cell: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None
        forward_params = inspect.signature(model.forward).parameters
        accepts_hidden = "hidden" in forward_params
        accepts_cell = "cell" in forward_params

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_progress):
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                forward_kwargs = {}
                if accepts_hidden and hidden is not None:
                    forward_kwargs["hidden"] = hidden
                if accepts_cell and cell is not None:
                    forward_kwargs["cell"] = cell

                outputs = model(inputs, **forward_kwargs)

                if isinstance(outputs, tuple):
                    logits, state = outputs

                    if isinstance(state, tuple):
                        hidden, cell = state
                    else:
                        hidden = state
                        cell = None
                else:
                    logits = outputs
                    hidden, cell = None, None

                if hidden is not None:
                    hidden = (
                        [h.detach() for h in hidden]
                        if isinstance(hidden, list)
                        else hidden.detach()
                    )
                if cell is not None:
                    cell = (
                        [c.detach() for c in cell]
                        if isinstance(cell, list)
                        else cell.detach()
                    )

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(
                        "Model produced NaN or Inf logits during evaluation"
                    )

                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_value = loss.item()
                total_valid_loss += loss_value

                context = {
                    "loss": loss_value,
                    "outputs": logits,
                    "targets": targets,
                }

                batch_metrics = {"loss": loss_value}
                for name in metrics_tracker.metric_names:
                    if hasattr(lm_metrics, name):
                        func = getattr(lm_metrics, name)
                        batch_metrics[name] = func(context)

                batch_size = inputs.size(0)
                metrics_tracker.update(batch_metrics, batch_size)

                if is_main and hasattr(valid_progress, "set_postfix"):
                    current_metrics = metrics_tracker.get_averages()
                    postfix = {
                        "Val Loss": f"{loss_value:.4f}",
                        "Avg Val Loss": f"{total_valid_loss / (batch_idx + 1):.4f}",
                    }
                    postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                    cast(tqdm, valid_progress).set_postfix(postfix)

        self.on_epoch_end(model, training=False)

        avg_valid_loss = total_valid_loss / len(valid_loader)

        if use_ddp:
            avg_valid_loss_tensor = torch.tensor(avg_valid_loss, device=device)
            dist.all_reduce(avg_valid_loss_tensor, op=dist.ReduceOp.AVG)
            avg_valid_loss = avg_valid_loss_tensor.item()

            avg_metrics = metrics_tracker.get_averages()
            for key in avg_metrics:
                metric_tensor = torch.tensor(avg_metrics[key], device=device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
                avg_metrics[key] = metric_tensor.item()
        else:
            avg_metrics = metrics_tracker.get_averages()

        return avg_valid_loss, avg_metrics

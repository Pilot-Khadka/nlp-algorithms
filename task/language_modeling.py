from typing import Dict, Any, Union, cast, Optional, Tuple, List
from torch import Tensor


import inspect
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

from task.base_task import BaseTask
import util.metric as lm_metrics


class LanguageModelingTask(BaseTask):
    @property
    def name(self):
        return "language_modeling"

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.vocab_size

    def get_loss_fn(self, pad_idx=0):
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

    def _is_ptb_iterator(self, loader) -> bool:
        return hasattr(loader, "reset") and hasattr(loader, "pos")

    def _reset_loader(self, loader) -> None:
        if self._is_ptb_iterator(loader):
            loader.reset()

    def _get_num_batches(self, loader) -> int:
        return len(loader)

    def on_epoch_start(self, model, training: bool = True) -> None:
        """
        Called at the start of each epoch.
        Resets model state for a fresh start on the data.
        """
        if hasattr(model, "reset_state"):
            model.reset_state()

    def on_epoch_end(self, model, training: bool = True) -> None:
        """Called at the end of each epoch."""
        pass

    def _detach_hidden(self, hidden):
        if hidden is None:
            return None
        if isinstance(hidden, Tensor):
            return hidden.detach()
        elif isinstance(hidden, list):
            return [self._detach_hidden(h) for h in hidden]
        elif isinstance(hidden, tuple):
            return tuple(self._detach_hidden(h) for h in hidden)
        return hidden

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

        is_ptb = self._is_ptb_iterator(train_loader)
        if use_ddp and not is_ptb:
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

        self.on_epoch_start(model, training=True)
        self._reset_loader(train_loader)

        total_train_loss = 0.0
        total_tokens = 0

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
            # For PTB iterator: x, y are (seq_len, batch_size) by default
            # For DataLoader: x, y are (batch_size, seq_len)
            inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

            hidden = self._detach_hidden(hidden)
            cell = self._detach_hidden(cell)

            optimizer.zero_grad()

            forward_kwargs = {}
            if accepts_hidden and hidden is not None:
                forward_kwargs["hidden"] = hidden
            if accepts_cell and cell is not None:
                forward_kwargs["cell"] = cell

            outputs = model(inputs, **forward_kwargs)

            if isinstance(outputs, tuple):
                logits, state = outputs

                # LSTM: (hidden, cell)
                if isinstance(state, tuple) and len(state) == 2:
                    hidden, cell = state
                # RNN / GRU: hidden only
                else:
                    hidden = state
                    cell = None
            else:
                logits = outputs
                hidden, cell = None, None

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            loss_value = loss.item()
            batch_tokens = targets.numel()
            total_train_loss += loss_value * batch_tokens
            total_tokens += batch_tokens

            if is_main and hasattr(train_progress, "set_postfix"):
                avg_loss = total_train_loss / total_tokens
                cast(tqdm, train_progress).set_postfix(
                    {
                        "Loss": f"{loss_value:.4f}",
                        "Avg Loss": f"{avg_loss:.4f}",
                        "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                    }
                )

        self.on_epoch_end(model, training=True)

        avg_train_loss = total_train_loss / total_tokens if total_tokens > 0 else 0.0

        if use_ddp:
            loss_tensor = torch.tensor([total_train_loss, total_tokens], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor[0] / loss_tensor[1]).item()

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
        model.eval()
        criterion = self.get_loss_fn()
        total_valid_loss = 0.0
        total_tokens = 0
        metrics_tracker.reset()

        self.on_epoch_start(model, training=False)
        self._reset_loader(valid_loader)

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

                    if isinstance(state, tuple) and len(state) == 2:
                        hidden, cell = state
                    else:
                        hidden = state
                        cell = None
                else:
                    logits = outputs
                    hidden, cell = None, None

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(
                        "Model produced NaN or Inf logits during evaluation"
                    )

                loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                loss_value = loss.item()

                batch_tokens = targets.numel()
                total_valid_loss += loss_value * batch_tokens
                total_tokens += batch_tokens

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

                metrics_tracker.update(batch_metrics, batch_tokens)

                if is_main and hasattr(valid_progress, "set_postfix"):
                    current_metrics = metrics_tracker.get_averages()
                    avg_loss = total_valid_loss / total_tokens
                    postfix = {
                        "Val Loss": f"{loss_value:.4f}",
                        "Avg Val Loss": f"{avg_loss:.4f}",
                        "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                    }
                    postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                    cast(tqdm, valid_progress).set_postfix(postfix)

        self.on_epoch_end(model, training=False)

        avg_valid_loss = total_valid_loss / total_tokens if total_tokens > 0 else 0.0

        if use_ddp:
            loss_tensor = torch.tensor([total_valid_loss, total_tokens], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_valid_loss = (loss_tensor[0] / loss_tensor[1]).item()

            avg_metrics = metrics_tracker.get_averages()
            for key in avg_metrics:
                metric_tensor = torch.tensor(avg_metrics[key], device=device)
                dist.all_reduce(metric_tensor, op=dist.ReduceOp.AVG)
                avg_metrics[key] = metric_tensor.item()
        else:
            avg_metrics = metrics_tracker.get_averages()

        return avg_valid_loss, avg_metrics

from typing import Dict, Any, Union, cast, Optional, Tuple, List
from torch import Tensor


import inspect
from tqdm import tqdm

import torch
import torch.distributed as dist

from training.base import BaseTrainer
import util.metric as lm_metrics
from engine.registry import register_trainer


@register_trainer("language_modeling")
class LanguageModelingTask(BaseTrainer):
    def on_epoch_start(self) -> None:
        """
        description: resets model state for a fresh start on the data.
        """
        if hasattr(self.model, "reset_state"):
            self.model.reset_state()

    def _is_iterator(self, loader) -> bool:
        return hasattr(loader, "reset") and hasattr(loader, "pos")

    def _reset_loader(self, loader) -> None:
        if self._is_iterator(loader):
            loader.reset()

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

    def train_one_epoch(self, epoch, total_epochs) -> float:
        self.model.train()

        if self.use_ddp:
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

        self._reset_loader(self.train_loader)
        total_train_loss = 0.0
        total_tokens = 0

        train_progress: Union[Any, tqdm]
        if self.is_main:
            train_progress = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Train]",
                leave=False,
            )
        else:
            train_progress = self.train_loader

        hidden: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None
        cell: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None

        forward_params = inspect.signature(self.model.forward).parameters
        accepts_hidden = "hidden" in forward_params
        accepts_cell = "cell" in forward_params

        for batch_idx, batch in enumerate(train_progress):
            # For PTB iterator: x, y are (seq_len, batch_size) by default
            # For DataLoader: x, y are (batch_size, seq_len)
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            hidden = self._detach_hidden(hidden)
            cell = self._detach_hidden(cell)

            self.optimizer.zero_grad()

            forward_kwargs = {}
            if accepts_hidden and hidden is not None:
                forward_kwargs["hidden"] = hidden
            if accepts_cell and cell is not None:
                forward_kwargs["cell"] = cell

            outputs = self.model(inputs, **forward_kwargs)

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

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )

            self.optimizer.step()

            loss_value = loss.item()
            batch_tokens = targets.numel()
            total_train_loss += loss_value * batch_tokens
            total_tokens += batch_tokens

            if self.is_main and hasattr(train_progress, "set_postfix"):
                avg_loss = total_train_loss / total_tokens
                cast(tqdm, train_progress).set_postfix(
                    {
                        "Loss": f"{loss_value:.4f}",
                        "Avg Loss": f"{avg_loss:.4f}",
                        "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                    }
                )

        avg_train_loss = total_train_loss / total_tokens if total_tokens > 0 else 0.0

        if self.use_ddp:
            loss_tensor = torch.tensor(
                [total_train_loss, total_tokens], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor[0] / loss_tensor[1]).item()

        return avg_train_loss

    def evaluate_one_epoch(self, epoch, total_epochs) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        total_valid_loss = 0.0
        total_tokens = 0
        self.metrics_tracker.reset()

        self.on_epoch_start()
        self._reset_loader(self.test_loader)

        valid_progress: Union[Any, tqdm]
        if self.is_main:
            valid_progress = tqdm(
                self.test_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Valid]",
                leave=False,
            )
        else:
            valid_progress = self.test_loader

        hidden: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None
        cell: Optional[Union[Tensor, List[Tensor], Tuple[Tensor, ...]]] = None

        forward_params = inspect.signature(self.model.forward).parameters
        accepts_hidden = "hidden" in forward_params
        accepts_cell = "cell" in forward_params

        all_logits = []
        all_targets = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_progress):
                inputs, targets = batch
                batch_size = inputs.size(0)
                batch_tokens = targets.numel()

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                forward_kwargs = {}
                if accepts_hidden and hidden is not None:
                    forward_kwargs["hidden"] = hidden
                if accepts_cell and cell is not None:
                    forward_kwargs["cell"] = cell

                outputs = self.model(inputs, **forward_kwargs)

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

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                loss_value = loss.item()

                total_valid_loss += loss_value * batch_size
                total_tokens += batch_tokens

                # invariant of sequence length
                all_logits.append(
                    logits.view(-1, logits.size(-1)).detach()
                )  # (batch*seq, vocab)
                all_targets.append(targets.view(-1).detach())  # (batch*seq,)

                if self.is_main and hasattr(valid_progress, "set_postfix"):
                    current_metrics = self.metrics_tracker.get_averages()
                    avg_loss = total_valid_loss / total_tokens
                    postfix = {
                        "Val Loss": f"{loss_value:.4f}",
                        "Avg Val Loss": f"{avg_loss:.4f}",
                        "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                    }
                    postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                    cast(tqdm, valid_progress).set_postfix(postfix)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        avg_valid_loss = total_valid_loss / total_tokens if total_tokens > 0 else 0.0

        if self.use_ddp:
            loss_tensor = torch.tensor(
                [total_valid_loss, float(total_tokens)], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_valid_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_valid_loss = (
                total_valid_loss / total_tokens if total_tokens > 0 else 0.0
            )

        if self.use_ddp:
            world_size = dist.get_world_size()

            logits_list = [torch.empty_like(all_logits) for _ in range(world_size)]
            targets_list = [torch.empty_like(all_targets) for _ in range(world_size)]

            dist.all_gather(logits_list, all_logits)
            dist.all_gather(targets_list, all_targets)

            all_logits = torch.cat(logits_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)

        context = {
            "loss": avg_valid_loss,
            "outputs": all_logits.cpu(),
            "targets": all_targets.cpu(),
        }

        avg_metrics = {"loss": avg_valid_loss}
        for name in self.metrics_tracker.metric_names:
            if hasattr(lm_metrics, name):
                avg_metrics[name] = getattr(lm_metrics, name)(context)

        self.metrics_tracker.update(avg_metrics)
        return avg_valid_loss, avg_metrics

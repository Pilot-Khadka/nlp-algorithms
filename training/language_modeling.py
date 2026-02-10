from typing import Dict, cast
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

    def _detach(self, x):
        if x is None:
            return None
        if isinstance(x, Tensor):
            return x.detach()
        if isinstance(x, (list, tuple)):
            return type(x)(self._detach(v) for v in x)
        return x

    def _prepare_forward_kwargs(self, accepts_hidden, accepts_cell, hidden, cell):
        kwargs = {}
        if accepts_hidden and hidden is not None:
            kwargs["hidden"] = hidden
        if accepts_cell and cell is not None:
            kwargs["cell"] = cell
        return kwargs

    def _get_forward_accepts(self):
        params = inspect.signature(self.model.forward).parameters
        return "hidden" in params, "cell" in params

    def _create_progress(self, loader, desc):
        if self.is_main:
            return tqdm(loader, desc=desc, leave=False)
        return loader

    def train_one_epoch(self, epoch, total_epochs) -> float:
        self.model.train()

        if self.use_ddp:
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

        self._reset_loader(self.train_loader)

        hidden = cell = None
        accepts_hidden, accepts_cell = self._get_forward_accepts()

        progress = self._create_progress(
            self.train_loader,
            f"Epoch {epoch + 1}/{total_epochs} [Train]",
        )

        train_loss = 0.0
        total_tokens = 0
        for inputs, targets in progress:
            # For PTB iterator: x, y are (seq_len, batch_size) by default
            # For DataLoader: x, y are (batch_size, seq_len)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            hidden = self._detach(hidden)
            cell = self._detach(cell)

            self.optimizer.zero_grad()

            outputs = self.model(
                inputs,
                **self._prepare_forward_kwargs(
                    accepts_hidden, accepts_cell, hidden, cell
                ),
            )

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

            batch_tokens = targets.numel()
            train_loss += loss.item()
            total_tokens += batch_tokens

            if self.is_main and hasattr(progress, "set_postfix"):
                avg_loss = train_loss / total_tokens
                cast(tqdm, progress).set_postfix(
                    {
                        "Avg Loss": f"{avg_loss:.4f}",
                        "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                    }
                )

        avg_train_loss = train_loss / total_tokens if total_tokens > 0 else 0.0

        if self.use_ddp:
            loss_tensor = torch.tensor([train_loss, total_tokens], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor[0] / loss_tensor[1]).item()

        return avg_train_loss

    def evaluate_one_epoch(self, epoch, total_epochs) -> tuple[float, Dict[str, float]]:
        self.model.eval()
        self.metrics_tracker.reset()

        self.on_epoch_start()
        self._reset_loader(self.test_loader)

        hidden = cell = None
        accepts_hidden, accepts_cell = self._get_forward_accepts()

        progress = self._create_progress(
            self.test_loader,
            f"Epoch {epoch + 1}/{total_epochs} [Valid]",
        )

        total_loss = 0.0
        total_tokens = 0

        all_logits = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_tokens = targets.numel()

                outputs = self.model(
                    inputs,
                    **self._prepare_forward_kwargs(
                        accepts_hidden, accepts_cell, hidden, cell
                    ),
                )

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

                total_loss += loss_value * batch_tokens
                total_tokens += batch_tokens

                # invariant of sequence length
                all_logits.append(
                    logits.view(-1, logits.size(-1)).detach().cpu()
                )  # (batch*seq, vocab)
                all_targets.append(targets.view(-1).detach().cpu())  # (batch*seq,)

                if self.is_main and hasattr(progress, "set_postfix"):
                    current_metrics = self.metrics_tracker.get_averages()
                    avg_loss = total_loss / total_tokens
                    postfix = {
                        "Val Loss": f"{loss_value:.4f}",
                        "Avg Val Loss": f"{avg_loss:.4f}",
                        "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                    }
                    postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                    cast(tqdm, progress).set_postfix(postfix)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        avg_valid_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

        if self.use_ddp:
            loss_tensor = torch.tensor(
                [total_loss, float(total_tokens)], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_valid_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_valid_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

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

        avg_metrics = {"loss": avg_valid_loss, "tokens": total_tokens}
        for name in self.metrics_tracker.metric_names:
            if hasattr(lm_metrics, name):
                avg_metrics[name] = getattr(lm_metrics, name)(context)

        self.metrics_tracker.update(avg_metrics)
        return avg_valid_loss, avg_metrics

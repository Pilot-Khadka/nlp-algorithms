from typing import Dict, Any, Union, cast, Optional, Tuple
from torch import Tensor

import inspect
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

from tasks.base_task import BaseTask
import utils.metrics as sa_metrics


class SentimentAnalysisTask(BaseTask):
    @property
    def name(self):
        return "sentiment_analysis"

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.num_classes

    def get_loss_fn(self, pad_idx=0):
        return nn.CrossEntropyLoss()

    def on_epoch_start(self, model, training: bool = True) -> None:
        if hasattr(model, "reset_state"):
            model.reset_state()

    def on_epoch_end(self, model, training: bool = True) -> None:
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

    def _reset_hidden_for_new_docs(
        self,
        hidden,
        is_first_chunk: Tensor,
        batch_size: int,
    ):
        """
           For samples that are starting a new document.

        Args:
        :hidden: Current hidden state (num_layers, batch, hidden_size) or tuple for LSTM
        :is_first_chunk: Boolean tensor (batch_size,) indicating new documents
        :batch_size: Current batch size
        """
        if hidden is None:
            return None

        if isinstance(hidden, tuple):
            # LSTM: (h, c)
            return tuple(
                self._reset_hidden_for_new_docs(h, is_first_chunk, batch_size)
                for h in hidden
            )

        # hidden shape: (num_layers, batch, hidden_size)
        if hidden.size(1) != batch_size:
            # batch size changed, return None to reinitialize
            return None

        mask = is_first_chunk.unsqueeze(0).unsqueeze(-1)
        hidden = hidden * (~mask).float()

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

        if use_ddp:
            if hasattr(train_loader, "sampler") and hasattr(
                train_loader.sampler, "set_epoch"
            ):
                train_loader.sampler.set_epoch(epoch)

        self.on_epoch_start(model, training=True)

        total_train_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        train_progress: Union[Any, tqdm]
        if is_main:
            train_progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Train]",
                leave=False,
            )
        else:
            train_progress = train_loader

        hidden: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None

        forward_params = inspect.signature(model.forward).parameters
        accepts_hidden = "hidden" in forward_params

        for batch_idx, batch in enumerate(train_progress):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            is_first_chunk = batch["is_first_chunk"].to(device)
            is_last_chunk = batch["is_last_chunk"].to(device)

            batch_size = input_ids.size(0)

            hidden = self._detach_hidden(hidden)
            hidden = self._reset_hidden_for_new_docs(hidden, is_first_chunk, batch_size)

            optimizer.zero_grad()

            forward_kwargs = {}
            if accepts_hidden and hidden is not None:
                forward_kwargs["hidden"] = hidden

            outputs = model(input_ids, **forward_kwargs)

            if isinstance(outputs, tuple):
                logits, new_hidden = outputs
                hidden = new_hidden
            else:
                logits = outputs
                hidden = None

            # For chunked sequences, we only compute loss on the last chunk
            # This is because the label applies to the full document
            last_chunk_mask = is_last_chunk

            if last_chunk_mask.any():
                last_chunk_logits = logits[last_chunk_mask]
                last_chunk_labels = labels[last_chunk_mask]

                loss = criterion(last_chunk_logits, last_chunk_labels)
                num_complete = last_chunk_mask.sum().item()
                predictions = torch.argmax(last_chunk_logits, dim=1)
                correct = (predictions == last_chunk_labels).sum().item()

                loss.backward()

                if grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=grad_clip
                    )

                optimizer.step()

                total_train_loss += loss.item() * num_complete
                total_samples += num_complete
                correct_predictions += correct
            else:
                # no complete documents in this batch, still need forward for hidden states
                # no loss computation or optimizer step
                pass

            if is_main and hasattr(train_progress, "set_postfix") and total_samples > 0:
                avg_loss = total_train_loss / total_samples
                avg_acc = correct_predictions / total_samples
                cast(tqdm, train_progress).set_postfix(
                    {
                        "Loss": f"{loss.item() if last_chunk_mask.any() else 0:.4f}",  # pyrefly: ignore
                        "Avg Loss": f"{avg_loss:.4f}",
                        "Acc": f"{avg_acc:.4f}",
                    }
                )

        self.on_epoch_end(model, training=True)

        avg_train_loss = total_train_loss / total_samples if total_samples > 0 else 0.0

        if use_ddp:
            loss_tensor = torch.tensor(
                [total_train_loss, float(total_samples)], device=device
            )
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
    ) -> Tuple[float, Dict[str, float]]:
        model.eval()
        criterion = self.get_loss_fn()

        total_valid_loss = 0.0
        total_samples = 0
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

        hidden: Optional[Union[Tensor, Tuple[Tensor, ...]]] = None

        forward_params = inspect.signature(model.forward).parameters
        accepts_hidden = "hidden" in forward_params

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_progress):
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                is_first_chunk = batch["is_first_chunk"].to(device)
                is_last_chunk = batch["is_last_chunk"].to(device)

                batch_size = input_ids.size(0)

                hidden = self._reset_hidden_for_new_docs(
                    hidden, is_first_chunk, batch_size
                )

                forward_kwargs = {}
                if accepts_hidden and hidden is not None:
                    forward_kwargs["hidden"] = hidden

                outputs = model(input_ids, **forward_kwargs)

                if isinstance(outputs, tuple):
                    logits, new_hidden = outputs
                    hidden = new_hidden
                else:
                    logits = outputs
                    hidden = None

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(
                        "Model produced NaN or Inf logits during evaluation"
                    )

                last_chunk_mask = is_last_chunk

                if last_chunk_mask.any():
                    last_chunk_logits = logits[last_chunk_mask]
                    last_chunk_labels = labels[last_chunk_mask]

                    loss = criterion(last_chunk_logits, last_chunk_labels)
                    loss_value = loss.item()

                    num_complete = last_chunk_mask.sum().item()
                    total_valid_loss += loss_value * num_complete
                    total_samples += num_complete

                    predictions = torch.argmax(last_chunk_logits, dim=1)

                    context = {
                        "loss": loss_value,
                        "outputs": last_chunk_logits,
                        "predictions": predictions,
                        "targets": last_chunk_labels,
                    }

                    batch_metrics = {"loss": loss_value}
                    for name in metrics_tracker.metric_names:
                        if hasattr(sa_metrics, name):
                            func = getattr(sa_metrics, name)
                            batch_metrics[name] = func(context)

                    metrics_tracker.update(batch_metrics, num_complete)

                    if is_main and hasattr(valid_progress, "set_postfix"):
                        current_metrics = metrics_tracker.get_averages()
                        avg_loss = total_valid_loss / total_samples
                        postfix = {
                            "Val Loss": f"{loss_value:.4f}",
                            "Avg Val Loss": f"{avg_loss:.4f}",
                        }
                        postfix.update(
                            {k: f"{v:.4f}" for k, v in current_metrics.items()}
                        )
                        cast(tqdm, valid_progress).set_postfix(postfix)

        self.on_epoch_end(model, training=False)

        avg_valid_loss = total_valid_loss / total_samples if total_samples > 0 else 0.0

        if use_ddp:
            loss_tensor = torch.tensor(
                [total_valid_loss, float(total_samples)], device=device
            )
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

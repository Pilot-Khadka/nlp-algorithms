from typing import Dict, Any, Union, cast, Tuple

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

        for batch in train_progress:
            inputs, labels, *others = batch
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            if logits.dim() == 3:
                logits = logits[:, -1, :]

            loss = criterion(logits, labels)

            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()

            total_train_loss += loss.detach() * batch_size
            total_samples += batch_size
            correct_predictions += correct

            if is_main and hasattr(train_progress, "set_postfix"):
                avg_loss = total_train_loss / total_samples
                avg_acc = correct_predictions / total_samples
                cast(tqdm, train_progress).set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Avg Loss": f"{avg_loss:.4f}",
                        "Acc": f"{avg_acc:.4f}",
                    }
                )

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

        valid_progress: Union[Any, tqdm]
        if is_main:
            valid_progress = tqdm(
                valid_loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [Valid]",
                leave=False,
            )
        else:
            valid_progress = valid_loader

        with torch.no_grad():
            for batch in valid_progress:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                batch_size = input_ids.size(0)

                outputs = model(input_ids)

                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(
                        "Model produced NaN or Inf logits during evaluation"
                    )

                loss = criterion(logits, labels)
                loss_value = loss.item()

                total_valid_loss += loss_value * batch_size
                total_samples += batch_size

                predictions = torch.argmax(logits, dim=1)

                context = {
                    "loss": loss_value,
                    "outputs": logits,
                    "predictions": predictions,
                    "targets": labels,
                }

                batch_metrics = {"loss": loss_value}
                for name in metrics_tracker.metric_names:
                    if hasattr(sa_metrics, name):
                        func = getattr(sa_metrics, name)
                        batch_metrics[name] = func(context)

                metrics_tracker.update(batch_metrics, batch_size)

                if is_main and hasattr(valid_progress, "set_postfix"):
                    current_metrics = metrics_tracker.get_averages()
                    avg_loss = total_valid_loss / total_samples
                    postfix = {
                        "Val Loss": f"{loss_value:.4f}",
                        "Avg Val Loss": f"{avg_loss:.4f}",
                    }
                    postfix.update({k: f"{v:.4f}" for k, v in current_metrics.items()})
                    cast(tqdm, valid_progress).set_postfix(postfix)

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

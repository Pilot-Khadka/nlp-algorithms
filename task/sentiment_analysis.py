from typing import Dict, Any, Union, cast, Tuple

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist

from task.base_task import BaseTask
import util.metric as sa_metrics


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

        total_train_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch in train_progress:
            inputs, labels, *others = batch
            batch_size = inputs.size(0)

            optimizer.zero_grad()
            outputs = model(inputs)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 3:
                logits = logits[:, -1, :]

            loss = criterion(logits, labels)
            loss.backward()

            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()

            total_train_loss += loss.item() * batch_size
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

        all_logits = []
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in valid_progress:
                inputs, labels, *others = batch
                batch_size = inputs.size(0)

                outputs = model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

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

                preds = torch.argmax(logits, dim=1)

                all_logits.append(logits.detach())
                all_predictions.append(preds.detach())
                all_targets.append(labels.detach())

                if is_main and hasattr(valid_progress, "set_postfix"):
                    avg_loss = total_valid_loss / total_samples
                    cast(tqdm, valid_progress).set_postfix(
                        {
                            "Val Loss": f"{loss_value:.4f}",
                            "Avg Val Loss": f"{avg_loss:.4f}",
                        }
                    )

        # concatenate per-rank
        all_logits = torch.cat(all_logits, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if use_ddp:
            loss_tensor = torch.tensor(
                [total_valid_loss, float(total_samples)], device=device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_valid_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_valid_loss = (
                total_valid_loss / total_samples if total_samples > 0 else 0.0
            )

        if use_ddp:
            world_size = dist.get_world_size()

            logits_list = [torch.empty_like(all_logits) for _ in range(world_size)]
            preds_list = [torch.empty_like(all_predictions) for _ in range(world_size)]
            targets_list = [torch.empty_like(all_targets) for _ in range(world_size)]

            dist.all_gather(logits_list, all_logits)
            dist.all_gather(preds_list, all_predictions)
            dist.all_gather(targets_list, all_targets)

            all_logits = torch.cat(logits_list, dim=0)
            all_predictions = torch.cat(preds_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)

        context = {
            "loss": avg_valid_loss,
            "outputs": all_logits.cpu(),
            "predictions": all_predictions.cpu(),
            "targets": all_targets.cpu(),
        }

        avg_metrics = {"loss": avg_valid_loss}
        for name in metrics_tracker.metric_names:
            if hasattr(sa_metrics, name):
                avg_metrics[name] = getattr(sa_metrics, name)(context)

        metrics_tracker.update(avg_metrics)
        return avg_valid_loss, avg_metrics

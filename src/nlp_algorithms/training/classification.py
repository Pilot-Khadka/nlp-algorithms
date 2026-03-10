from typing import Union, Any, Tuple, Dict, cast

import time
from tqdm import tqdm

import torch
import torch.distributed as dist

from .base import BaseTrainer
from nlp_algorithms.util.metric import accuracy, precision, recall, f1_score
from nlp_algorithms.engine.registry import register_trainer


@register_trainer("classification")
class ClassificationTrainer(BaseTrainer):
    def train_one_epoch(self, epoch, total_epochs) -> tuple[float, float]:
        self.model.train()
        start_time = time.perf_counter()

        if self.use_ddp:
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

        total_train_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        train_progress: Union[Any, tqdm]
        if self.is_main:
            if self.config.show_progress:
                train_progress = tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch + 1}/{total_epochs} [Train]",
                    leave=False,
                    ncols=120,
                )
            else:
                print(f"Training epoch {epoch + 1}/{total_epochs}...")
                train_progress = self.train_loader
        else:
            train_progress = self.train_loader

        total_train_loss = 0.0
        total_samples = 0
        correct_predictions = 0

        for batch in train_progress:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            inputs = batch["input_ids"]
            labels = batch["labels"]

            assert labels.dtype == torch.long, labels.dtype

            batch_size = inputs.size(0)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            if logits.dim() == 3:
                logits = logits[:, -1, :]

            loss = self.criterion(logits, labels)
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )

            self.optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum().item()

            total_train_loss += loss.item() * batch_size
            total_samples += batch_size
            correct_predictions += correct

            if (
                self.is_main
                and self.config.show_progress
                and hasattr(train_progress, "set_postfix")
            ):
                avg_loss = total_train_loss / total_samples
                avg_acc = correct_predictions / total_samples
                short_metrics = self._tqdm_format_metrics(
                    {"Loss": loss.item(), "AvgLoss": avg_loss, "Acc": avg_acc}
                )
                cast(tqdm, train_progress).set_postfix_str(short_metrics)

        avg_train_loss = total_train_loss / total_samples if total_samples > 0 else 0.0

        if self.use_ddp:
            loss_tensor = torch.tensor(
                [total_train_loss, float(total_samples)], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_train_loss = (loss_tensor[0] / loss_tensor[1]).item()

        train_time = time.perf_counter() - start_time

        return avg_train_loss, train_time

    def evaluate_one_epoch(
        self,
        epoch,
        total_epochs,
    ) -> Tuple[float, float, Dict[str, float]]:
        self.model.eval()
        start_time = time.perf_counter()

        total_valid_loss = 0.0
        total_samples = 0

        valid_progress: Union[Any, tqdm]
        if self.is_main:
            if self.config.show_progress:
                valid_progress = tqdm(
                    self.test_loader,
                    desc=f"Epoch {epoch + 1}/{total_epochs} [Valid]",
                    leave=False,
                )
            else:
                print(f"Evaluating epoch {epoch + 1}/{total_epochs}...")
                valid_progress = self.test_loader
        else:
            valid_progress = self.test_loader

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in valid_progress:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                inputs = batch["input_ids"]
                labels = batch["labels"]
                batch_size = inputs.size(0)

                outputs = self.model(inputs)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(
                        "Model produced NaN or Inf logits during evaluation"
                    )

                loss = self.criterion(logits, labels)
                loss_value = loss.item()

                total_valid_loss += loss_value * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=1)

                # all_logits.append(logits.detach().cpu())
                all_predictions.append(preds.detach().cpu())
                all_targets.append(labels.detach().cpu())

                if (
                    self.is_main
                    and self.config.show_progress
                    and hasattr(valid_progress, "set_postfix")
                ):
                    avg_loss = total_valid_loss / total_samples
                    short_metrics = self._tqdm_format_metrics(
                        {"ValLoss": loss_value, "AvgValLoss": avg_loss}
                    )
                    cast(tqdm, valid_progress).set_postfix_str(short_metrics)

        # concatenate per-rank
        # all_logits = torch.cat(all_logits, dim=0)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if self.use_ddp:
            loss_tensor = torch.tensor(
                [total_valid_loss, float(total_samples)], device=self.device
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_valid_loss = (loss_tensor[0] / loss_tensor[1]).item()
        else:
            avg_valid_loss = (
                total_valid_loss / total_samples if total_samples > 0 else 0.0
            )

        if self.use_ddp:
            world_size = dist.get_world_size()

            # logits_list = [torch.empty_like(all_logits) for _ in range(world_size)]
            preds_list = [torch.empty_like(all_predictions) for _ in range(world_size)]
            targets_list = [torch.empty_like(all_targets) for _ in range(world_size)]

            # dist.all_gather(logits_list, all_logits)
            dist.all_gather(preds_list, all_predictions)
            dist.all_gather(targets_list, all_targets)

            # all_logits = torch.cat(logits_list, dim=0)
            all_predictions = torch.cat(preds_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)

        preds = all_predictions.cpu()
        tgts = all_targets.cpu()
        num_classes = self.config.dataset.num_class
        avg_metrics = {
            "accuracy": accuracy(preds, tgts),
            "precision": precision(preds, tgts, num_classes),
            "recall": recall(preds, tgts, num_classes),
            "f1": f1_score(preds, tgts, num_classes),
        }

        val_time = time.perf_counter() - start_time
        return avg_valid_loss, val_time, avg_metrics

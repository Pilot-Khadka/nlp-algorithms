from typing import Union, Any, cast

import time
from tqdm import tqdm

import torch
import torch.distributed as dist

from .base import BaseTrainer
from nlp_algorithms.util.metric import accuracy, precision, recall, f1_score
from nlp_algorithms.engine.registry import register_trainer


@register_trainer("classification")
class ClassificationTrainer(BaseTrainer):
    def __init__(self, config, builder, gpu_id: int = 0, use_ddp: bool = False):
        super().__init__(config, builder, gpu_id=gpu_id, use_ddp=use_ddp)
        # bert fine-tuning uses the pretraining schedule, the rnns keep a fixed lr
        self.lr_scheduler = None
        if config.model.name == "bert":
            self.lr_scheduler = self.warmup_linear_decay()

    def _forward(self, batch):
        if self.config.model.name == "bert":
            # padding is masked out of the attention
            return self.model(
                batch["input_ids"], attention_mask=batch["attention_mask"]
            )
        return self.model(batch["input_ids"])

    def train_one_epoch(self, epoch, total_epochs) -> tuple[float, float]:
        self.model.train()
        start_time = time.perf_counter()

        if self.use_ddp:
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

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
                train_progress = self.train_loader
        else:
            train_progress = self.train_loader

        # [loss sum, samples, correct], summed on the gpu to avoid a sync every step
        totals = torch.zeros(3, dtype=torch.float64, device=self.device)
        num_steps = len(self.train_loader)

        for step, batch in enumerate(train_progress, start=1):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            inputs = batch["input_ids"]
            labels = batch["labels"]

            assert labels.dtype == torch.long, labels.dtype

            batch_size = inputs.size(0)

            self.optimizer.zero_grad()
            with self.autocast():
                outputs = self._forward(batch)

                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                if logits.dim() == 3:
                    logits = logits[:, -1, :]

                loss = self.criterion(logits, labels)
            self.backward_step(loss)
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == labels).sum()
            totals += torch.stack(
                [loss.detach() * batch_size, labels.new_tensor(batch_size), correct]
            ).double()

            if self.should_log_step(step, num_steps):
                loss_sum, samples, correct_sum = totals.tolist()
                self.log_step(
                    epoch,
                    step,
                    num_steps,
                    start_time,
                    {"loss": loss_sum / samples, "acc": correct_sum / samples},
                )

            if (
                self.is_main
                and self.config.show_progress
                and hasattr(train_progress, "set_postfix")
                and step % (self.log_every or 50) == 0
            ):
                loss_sum, samples, correct_sum = totals.tolist()
                short_metrics = self._tqdm_format_metrics(
                    {"Loss": loss_sum / samples, "Acc": correct_sum / samples}
                )
                cast(tqdm, train_progress).set_postfix_str(short_metrics)

        if self.use_ddp:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_sum, samples = totals[:2].tolist()
        avg_train_loss = loss_sum / samples if samples > 0 else 0.0

        train_time = time.perf_counter() - start_time

        return avg_train_loss, train_time

    def evaluate_one_epoch(
        self,
        epoch,
        total_epochs,
    ) -> tuple[float, float, dict[str, float]]:
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
                valid_progress = self.test_loader
        else:
            valid_progress = self.test_loader

        all_predictions = []
        all_targets = []
        # checked once after the loop, a check per batch waits for the gpu every time
        non_finite = torch.zeros((), dtype=torch.bool, device=self.device)
        loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)

        with torch.no_grad():
            for batch in valid_progress:
                batch = {
                    k: v.to(self.device, non_blocking=True) for k, v in batch.items()
                }

                inputs = batch["input_ids"]
                labels = batch["labels"]
                batch_size = inputs.size(0)

                with self.autocast():
                    outputs = self._forward(batch)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs

                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                logits = logits.float()
                non_finite |= ~torch.isfinite(logits).all()

                loss = self.criterion(logits, labels)
                loss_sum += loss.detach().double() * batch_size
                total_samples += batch_size

                preds = torch.argmax(logits, dim=1)

                # all_logits.append(logits.detach().cpu())
                all_predictions.append(preds.detach())
                all_targets.append(labels.detach())

        if non_finite.item():
            raise ValueError("Model produced NaN or Inf logits during evaluation")
        total_valid_loss = loss_sum.item()

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

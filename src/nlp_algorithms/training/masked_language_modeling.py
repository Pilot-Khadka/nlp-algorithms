from typing import cast

import time
from tqdm import tqdm

import torch
import torch.distributed as dist

from .base import BaseTrainer
from nlp_algorithms.util.metric import perplexity
from nlp_algorithms.engine.registry import register_trainer


@register_trainer("masked_language_modeling")
class MaskedLanguageModelingTrainer(BaseTrainer):
    def __init__(self, config, builder, gpu_id: int = 0, use_ddp: bool = False):
        super().__init__(config, builder, gpu_id=gpu_id, use_ddp=use_ddp)
        self.lr_scheduler = self.warmup_linear_decay()

    def _forward(self, batch):
        mlm_logits, nsp_logits = self.model(
            batch["input_ids"],
            token_type_ids=batch["token_type_ids"],
            attention_mask=batch["attention_mask"],
        )
        mlm_loss = self.criterion(
            mlm_logits.view(-1, mlm_logits.size(-1)), batch["mlm_labels"].view(-1)
        )
        nsp_loss = self.criterion(nsp_logits, batch["nsp_labels"])
        return mlm_logits, nsp_logits, mlm_loss, nsp_loss

    def _progress(self, epoch, total_epochs, loader, split):
        if self.is_main and self.config.show_progress:
            return tqdm(
                loader,
                desc=f"Epoch {epoch + 1}/{total_epochs} [{split}]",
                leave=False,
                ncols=120,
            )
        return loader

    def _reduce(self, totals):
        # [mlm loss sum, masked tokens, mlm correct, nsp loss sum, pairs, nsp correct]
        totals = totals.clone()
        if self.use_ddp:
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        mlm_loss, tokens, mlm_correct, nsp_loss, pairs, nsp_correct = totals.tolist()
        tokens, pairs = max(tokens, 1), max(pairs, 1)
        return {
            "mlm_loss": mlm_loss / tokens,
            "mlm_accuracy": mlm_correct / tokens,
            "nsp_loss": nsp_loss / pairs,
            "nsp_accuracy": nsp_correct / pairs,
        }

    def _running(self, totals):
        # this rank only, for the step lines
        mlm_loss, tokens, mlm_correct, nsp_loss, pairs, _ = totals.tolist()
        tokens, pairs = max(tokens, 1), max(pairs, 1)
        return {
            "mlm_loss": mlm_loss / tokens,
            "mlm_acc": mlm_correct / tokens,
            "nsp_loss": nsp_loss / pairs,
        }

    def _run_epoch(self, epoch, total_epochs, train):
        loader = self.train_loader if train else self.test_loader
        progress = self._progress(
            epoch, total_epochs, loader, "Train" if train else "Valid"
        )
        # summed on the gpu, .item() every step would wait for the gpu each time
        totals = torch.zeros(6, dtype=torch.float64, device=self.device)
        start_time = time.perf_counter()

        for step, batch in enumerate(progress, start=1):
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}

            with torch.set_grad_enabled(train), self.autocast():
                mlm_logits, nsp_logits, mlm_loss, nsp_loss = self._forward(batch)

            if train:
                self.optimizer.zero_grad()
                self.backward_step(mlm_loss + nsp_loss)
                self.lr_scheduler.step()

            masked = batch["mlm_labels"] != -100
            tokens = masked.sum()
            pairs = (batch["nsp_labels"] >= 0).sum()
            mlm_correct = ((mlm_logits.argmax(-1) == batch["mlm_labels"]) & masked).sum()
            nsp_correct = (nsp_logits.argmax(-1) == batch["nsp_labels"]).sum()
            totals += torch.stack(
                [
                    mlm_loss.detach() * tokens,
                    tokens,
                    mlm_correct,
                    nsp_loss.detach() * pairs,
                    pairs,
                    nsp_correct,
                ]
            ).double()

            if train and self.should_log_step(step, len(loader)):
                self.log_step(
                    epoch, step, len(loader), start_time, self._running(totals)
                )

            if (
                self.is_main
                and self.config.show_progress
                and hasattr(progress, "set_postfix")
                and step % (self.log_every or 50) == 0
            ):
                running = self._running(totals)
                cast(tqdm, progress).set_postfix_str(
                    self._tqdm_format_metrics(
                        {"MLM": running["mlm_loss"], "NSP": running["nsp_loss"]}
                    )
                )

        return self._reduce(totals)

    def train_one_epoch(self, epoch, total_epochs) -> tuple[float, float]:
        self.model.train()
        start_time = time.perf_counter()

        if self.use_ddp and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)

        metrics = self._run_epoch(epoch, total_epochs, train=True)
        return metrics["mlm_loss"] + metrics[
            "nsp_loss"
        ], time.perf_counter() - start_time

    def evaluate_one_epoch(
        self, epoch, total_epochs
    ) -> tuple[float, float, dict[str, float]]:
        self.model.eval()
        start_time = time.perf_counter()

        metrics = self._run_epoch(epoch, total_epochs, train=False)
        metrics["mlm_perplexity"] = perplexity(metrics["mlm_loss"])
        valid_loss = metrics["mlm_loss"] + metrics["nsp_loss"]
        return valid_loss, time.perf_counter() - start_time, metrics

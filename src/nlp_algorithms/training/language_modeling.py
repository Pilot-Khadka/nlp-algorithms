from typing import cast, Union, Any
from torch import Tensor


import time
import inspect
from tqdm import tqdm

import torch
import torch.distributed as dist


from nlp_algorithms.util.metric import perplexity
from nlp_algorithms.training.base import BaseTrainer
from nlp_algorithms.engine.registry import register_trainer


@register_trainer("language_modeling")
class LanguageModelingTask(BaseTrainer):
    def _detach(self, x):
        if x is None:
            return None
        if isinstance(x, Tensor):
            return x.detach()
        if isinstance(x, (list, tuple)):
            return type(x)(self._detach(v) for v in x)
        return x

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _prepare_forward_kwargs(self, accepts_hidden, hidden):
        kwargs = {}
        if accepts_hidden and hidden is not None:
            kwargs["hidden"] = hidden
        return kwargs

    def _get_forward_accepts(self):
        model = self._unwrap_model()
        params = inspect.signature(model.encoder.forward).parameters
        return "hidden" in params

    def _create_progress(self, loader, desc):
        if self.is_main:
            return tqdm(loader, desc=desc, leave=False)
        return loader

    def _unpack_outputs(self, outputs):
        """
        outputs: (logits, state)
        state may be:
            - tensor (GRU/RNN)
            - (h, c) tuple (LSTM)
        """
        if not isinstance(outputs, tuple):
            return outputs, None

        logits, state = outputs

        # LSTM
        if isinstance(state, tuple) and len(state) == 2:
            return logits, state  # FULL tuple (h, c)

        # GRU / RNN
        return logits, state

    def train_one_epoch(self, epoch, total_epochs) -> tuple[float, float]:
        self.model.train()
        start_time = time.perf_counter()

        if self.use_ddp:
            if hasattr(self.train_loader, "sampler") and hasattr(
                self.train_loader.sampler, "set_epoch"
            ):
                self.train_loader.sampler.set_epoch(epoch)

        hidden = None
        accepts_hidden = self._get_forward_accepts()

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

        # summed on the gpu, .item() every step would wait for the gpu each time
        total_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        total_tokens = 0
        num_steps = len(self.train_loader)
        for step, (inputs, targets) in enumerate(train_progress, start=1):
            # For PTB iterator: x, y are (seq_len, batch_size) by default
            # For DataLoader: x, y are (batch_size, seq_len)
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            hidden = self._detach(hidden)

            self.optimizer.zero_grad()

            with self.autocast():
                outputs = self.model(
                    inputs,
                    **self._prepare_forward_kwargs(accepts_hidden, hidden),
                )

                logits, hidden = self._unpack_outputs(outputs)
                # CE mean reduced
                loss = self.criterion(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                )
            self.backward_step(loss)

            batch_tokens = targets.numel()
            # convert mean loss into token weighted sum
            total_loss_sum += loss.detach().double() * batch_tokens
            total_tokens += batch_tokens

            if self.should_log_step(step, num_steps):
                avg_loss = total_loss_sum.item() / total_tokens
                self.log_step(
                    epoch,
                    step,
                    num_steps,
                    start_time,
                    {"loss": avg_loss, "ppl": perplexity(loss=avg_loss)},
                )

            if (
                self.is_main
                and self.config.show_progress
                and hasattr(train_progress, "set_postfix")
                and step % (self.log_every or 50) == 0
            ):
                avg_loss = total_loss_sum.item() / total_tokens
                short_metrics = self._tqdm_format_metrics(
                    {"loss": avg_loss, "ppl": perplexity(loss=avg_loss)}
                )
                cast(tqdm, train_progress).set_postfix_str(short_metrics)

        total_loss_sum = total_loss_sum.item()
        if self.use_ddp:
            t = torch.tensor([total_loss_sum, total_tokens], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss_sum, total_tokens = t.tolist()

        train_time = time.perf_counter() - start_time
        return total_loss_sum / total_tokens, train_time

    def evaluate_one_epoch(
        self, epoch, total_epochs
    ) -> tuple[float, float, dict[str, float]]:
        self.model.eval()
        start_time = time.perf_counter()

        hidden = None
        accepts_hidden = self._get_forward_accepts()  # single bool

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

        total_loss_sum = torch.zeros((), dtype=torch.float64, device=self.device)
        total_tokens = 0
        # checked once after the loop, a check per batch waits for the gpu every time
        non_finite = torch.zeros((), dtype=torch.bool, device=self.device)

        with torch.no_grad():
            for inputs, targets in valid_progress:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                hidden = self._detach(hidden)

                with self.autocast():
                    outputs = self.model(
                        inputs, **self._prepare_forward_kwargs(accepts_hidden, hidden)
                    )
                logits, hidden = self._unpack_outputs(outputs)
                logits = logits.float()
                non_finite |= ~torch.isfinite(logits).all()

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                batch_tokens = targets.numel()

                total_loss_sum += loss.detach().double() * batch_tokens
                total_tokens += batch_tokens

        if non_finite.item():
            raise ValueError("Model produced NaN or Inf logits during evaluation")
        total_loss_sum = total_loss_sum.item()

        if self.use_ddp:
            t = torch.tensor([total_loss_sum, float(total_tokens)], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss_sum, total_tokens = t.tolist()

        avg_loss = total_loss_sum / total_tokens
        avg_metrics = {"perplexity": perplexity(loss=avg_loss)}

        val_time = time.perf_counter() - start_time
        return avg_loss, val_time, avg_metrics

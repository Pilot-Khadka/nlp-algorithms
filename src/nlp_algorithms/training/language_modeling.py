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

    def _prepare_forward_kwargs(self, accepts_hidden, hidden):
        kwargs = {}
        if accepts_hidden and hidden is not None:
            kwargs["hidden"] = hidden
        return kwargs

    def _get_forward_accepts(self):
        params = inspect.signature(self.model.encoder.forward).parameters
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
                print(f"Training epoch {epoch + 1}/{total_epochs}...")
                train_progress = self.train_loader
        else:
            train_progress = self.train_loader

        total_loss_sum = 0.0
        total_tokens = 0
        for inputs, targets in train_progress:
            # For PTB iterator: x, y are (seq_len, batch_size) by default
            # For DataLoader: x, y are (batch_size, seq_len)
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            hidden = self._detach(hidden)

            self.optimizer.zero_grad()

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
            loss.backward()

            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )

            self.optimizer.step()

            batch_tokens = targets.numel()
            # convert mean loss into token weighted sum
            loss_value = loss.item()
            total_loss_sum += loss_value * batch_tokens
            total_tokens += batch_tokens

            if (
                self.is_main
                and self.config.show_progress
                and hasattr(train_progress, "set_postfix")
            ):
                short_metrics = self._tqdm_format_metrics(
                    {"Val loss": loss_value, "ppl": perplexity(loss=loss)}
                )
                cast(tqdm, train_progress).set_postfix_str(short_metrics)

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
                print(f"Evaluating epoch {epoch + 1}/{total_epochs}...")
                valid_progress = self.test_loader
        else:
            valid_progress = self.test_loader

        total_loss_sum = 0.0
        total_tokens = 0
        all_logits = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in valid_progress:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                hidden = self._detach(hidden)  # handles GRU tensor and LSTM (h,c) tuple

                outputs = self.model(
                    inputs,
                    **self._prepare_forward_kwargs(
                        accepts_hidden, hidden
                    ),  # correct arity
                )

                logits, hidden = self._unpack_outputs(outputs)  # always a 2-tuple

                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    raise ValueError(
                        "Model produced NaN or Inf logits during evaluation"
                    )

                loss = self.criterion(
                    logits.view(-1, logits.size(-1)), targets.view(-1)
                )
                loss_value = loss.item()
                batch_tokens = targets.numel()

                total_loss_sum += loss_value * batch_tokens
                total_tokens += batch_tokens

                all_logits.append(logits.view(-1, logits.size(-1)).detach().cpu())
                all_targets.append(targets.view(-1).detach().cpu())

                if (
                    self.is_main
                    and self.config.show_progress
                    and hasattr(valid_progress, "set_postfix")
                ):
                    avg_loss = total_loss_sum / total_tokens
                    cast(tqdm, valid_progress).set_postfix(
                        {
                            "Val Loss": f"{loss_value:.4f}",
                            "PPL": f"{torch.exp(torch.tensor(avg_loss)).item():.2f}",
                        }
                    )

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        if self.use_ddp:
            t = torch.tensor([total_loss_sum, float(total_tokens)], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss_sum, total_tokens = t.tolist()

            world_size = dist.get_world_size()
            logits_list = [torch.empty_like(all_logits) for _ in range(world_size)]
            targets_list = [torch.empty_like(all_targets) for _ in range(world_size)]
            dist.all_gather(logits_list, all_logits)
            dist.all_gather(targets_list, all_targets)
            all_logits = torch.cat(logits_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)

        avg_loss = total_loss_sum / total_tokens
        avg_metrics = {"perplexity": perplexity(loss=avg_loss)}

        val_time = time.perf_counter() - start_time
        return avg_loss, val_time, avg_metrics

import torch
import torch.nn as nn


from tasks.base_task import BaseTask
import utils.metrics as lm_metrics


def unwrap_model(model: nn.Module) -> nn.Module:
    from torch.nn.parallel import DistributedDataParallel as DDP

    """Unwrap model from DDP/DataParallel wrapper if necessary."""
    if isinstance(model, (DDP, nn.DataParallel)):
        return model.module
    return model


class LanguageModelingTask(BaseTask):
    @property
    def name(self):
        return "language_modeling"

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.vocab_size

    def get_loss_fn(self, pad_idx=0):
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

    def on_epoch_start(self, model: nn.Module, training: bool = True) -> None:
        """
        called at the start of each epoch.
        rests model state for a fresh start on the data.
        """
        base_model = unwrap_model(model)
        if hasattr(base_model, "reset_state"):
            base_model.reset_state()

    def on_epoch_end(self, model: nn.Module, training: bool = True) -> None:
        """Called at the end of each epoch. Optional cleanup."""
        pass

    def train_step(
        self,
        batch,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        grad_clip,
        device: torch.device,
    ):
        base_model = unwrap_model(model)
        if hasattr(base_model, "detach_state"):
            base_model.detach_state()

        criterion = self.get_loss_fn()
        inputs, targets = batch  # shape: [batch_size, seq_len]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: [batch_size, seq_len, vocab_size]

        # if the model returns (output, hidden) or (output, (h, c))
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()

        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        return loss.item()

    def eval_step(self, batch, model, device, metrics_list):
        base_model = unwrap_model(model)

        if hasattr(base_model, "detach_state"):
            base_model.detach_state()

        criterion = self.get_loss_fn()
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                raise ValueError("Model produced NaN or Inf logits during evaluation")

            loss = criterion(
                outputs.view(-1, outputs.size(-1)), targets.view(-1)
            ).item()

            context = {
                "loss": loss,
                "outputs": outputs,
                "targets": targets,
            }

            metrics = {"loss": loss}
            for name in metrics_list:
                if hasattr(lm_metrics, name):
                    func = getattr(lm_metrics, name)
                    metrics[name] = func(context)

        return loss, metrics

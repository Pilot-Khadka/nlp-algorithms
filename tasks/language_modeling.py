import torch
import torch.nn as nn


from tasks.base_task import BaseTask
import utils.metrics as lm_metrics


class LanguageModelingTask(BaseTask):
    @property
    def name(self):
        return "language_modeling"

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.vocab_size

    def get_loss_fn(self, pad_idx=0):
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

    def train_step(self, batch, model, optimizer, device):
        criterion = self.get_loss_fn()
        inputs, targets = batch  # shape: [batch_size, seq_len]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: [batch_size, seq_len, vocab_size]

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(self, batch, model, device, metrics_list):
        criterion = self.get_loss_fn()
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            outputs = model(inputs)

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

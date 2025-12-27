import torch
import torch.nn as nn


from tasks.base_task import BaseTask
import utils.metrics as lm_metrics


class LanguageModelingTask(BaseTask):
    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.vocab_size

    def get_loss_fn(self):
        return nn.CrossEntropyLoss()

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
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))

            metrics = self.compute_metrics(outputs, targets, metrics_list)
        return loss, metrics

    def compute_metrics(self, outputs, targets, metrics_list):
        computed = {}
        for metric_name in metrics_list:
            if hasattr(lm_metrics, metric_name):
                func = getattr(lm_metrics, metric_name)
                result = func(outputs, targets, computed)
                computed[metric_name] = result
        return computed

    @property
    def name(self):
        return "language_modeling"

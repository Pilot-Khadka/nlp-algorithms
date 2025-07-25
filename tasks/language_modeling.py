import torch
import torch.nn as nn
from tasks.base_task import BaseTask


class LanguageModelingTask(BaseTask):
    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.vocab_size

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def train_step(self, batch, model, criterion, optimizer, device):
        inputs, targets = batch  # shape: [batch_size, seq_len]
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: [batch_size, seq_len, vocab_size]

        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(self, batch, model, criterion, device):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(
                outputs.view(-1, outputs.size(-1)), targets.view(-1))
        return loss.item()

    def compute_metrics(self, outputs, targets):
        return {}

import torch
import torch.nn as nn
from tasks.base_task import BaseTask


class SentimentAnalysisTask(BaseTask):
    def name(self):
        return "sentiment_analysis"

    def train_step(self, batch, model, criterion, optimizer, device):
        inputs, labels = batch  # inputs: [batch, seq_len], labels: [batch]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: [batch, num_classes]

        # for sentiment analysis, take only the last time step
        outputs = outputs[:, -1, :]  # shape: [batch, num_classes]

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def get_loss_function(self):
        return nn.CrossEntropyLoss()

    def eval_step(self, batch, model, criterion, device):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model(inputs)
            outputs = outputs[:, -1, :]  # shape: [batch, num_classes]
            loss = criterion(outputs, labels)
        return loss.item()

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.num_classes

    def compute_metrics(self, outputs, targets):
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        return {"accuracy": accuracy}

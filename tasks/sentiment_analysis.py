from tasks.base_task import BaseTask
import torch


class SentimentAnalysisTask(BaseTask):
    def train_step(self, batch, model, criterion, optimizer, device):
        inputs, labels = batch  # inputs: [batch, seq_len], labels: [batch]
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: [batch, num_classes]
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_step(self, batch, model, criterion, device):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        return loss.item()

    def compute_metrics(self, outputs, targets):
        predictions = torch.argmax(outputs, dim=1)
        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total
        return {"accuracy": accuracy}

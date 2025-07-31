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

            metrics = self.compute_metrics(outputs, labels)

        return loss.item(), metrics

    def get_output_dim(self, dataset_bundle):
        return dataset_bundle.num_classes

    def compute_metrics(self, outputs, targets):
        predictions = torch.argmax(outputs, dim=1)

        correct = (predictions == targets).sum().item()
        total = targets.size(0)
        accuracy = correct / total

        pred_np = predictions.cpu().numpy()
        target_np = targets.cpu().numpy()

        num_classes = outputs.shape[1]

        # per-class metrics
        precision_per_class = []
        recall_per_class = []
        f1_per_class = []
        support_per_class = []

        for class_idx in range(num_classes):
            # True positives, false positives, false negatives for this class
            tp = ((pred_np == class_idx) & (target_np == class_idx)).sum()
            fp = ((pred_np == class_idx) & (target_np != class_idx)).sum()
            fn = ((pred_np != class_idx) & (target_np == class_idx)).sum()

            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # F1 = 2 * (precision * recall) / (precision + recall)
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Support : number of true instances of this class
            support = (target_np == class_idx).sum()

            precision_per_class.append(precision)
            recall_per_class.append(recall)
            f1_per_class.append(f1)
            support_per_class.append(support)

        # Macro averages (unweighted mean across classes)
        precision_macro = sum(precision_per_class) / num_classes
        recall_macro = sum(recall_per_class) / num_classes
        f1_macro = sum(f1_per_class) / num_classes

        # Weighted averages (weighted by support)
        total_support = sum(support_per_class)
        if total_support > 0:
            precision_weighted = (
                sum(p * s for p, s in zip(precision_per_class, support_per_class))
                / total_support
            )
            recall_weighted = (
                sum(r * s for r, s in zip(recall_per_class, support_per_class))
                / total_support
            )
            f1_weighted = (
                sum(f * s for f, s in zip(f1_per_class, support_per_class))
                / total_support
            )
        else:
            precision_weighted = recall_weighted = f1_weighted = 0.0

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "batch_size": total,  # Track batch size for averaging
        }

        # Add per-class metrics
        for i in range(num_classes):
            metrics[f"precision_class_{i}"] = precision_per_class[i]
            metrics[f"recall_class_{i}"] = recall_per_class[i]
            metrics[f"f1_class_{i}"] = f1_per_class[i]

        return metrics

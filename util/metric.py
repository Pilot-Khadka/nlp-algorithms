from typing import Dict, List, Optional, Any, TypedDict


import torch
from torch import Tensor

from collections import defaultdict


class MetricContext(TypedDict):
    loss: float
    outputs: Tensor
    predictions: Tensor
    targets: Tensor


class MetricsTracker:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()

    def reset(self) -> None:
        self.batch_metrics: List[Dict[str, float]] = []
        self.accumulated: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)

    def update(self, metrics: Dict[str, float]) -> None:
        self.batch_metrics.append(metrics)

        for name, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.accumulated[name] += value
            self.counts[name] += 1

    def get_averages(self) -> Dict[str, float]:
        if not self.accumulated:
            return {}

        averages = {}
        for metric_name in self.accumulated:
            if self.counts[metric_name] > 0:
                averages[metric_name] = (
                    self.accumulated[metric_name] / self.counts[metric_name]
                )

        return averages

    def get_current_batch_metrics(self) -> Dict[str, float]:
        if not self.batch_metrics:
            return {}
        return self.batch_metrics[-1]

    def format_metrics(
        self, metrics: Optional[Dict[str, float]] = None, prefix: str = ""
    ) -> str:
        if metrics is None:
            metrics = self.get_averages()

        if not metrics:
            return f"{prefix}No metrics available"

        formatted: list[str] = []
        for name, value in sorted(metrics.items()):
            formatted.append(f"{name}: {value:.4f}")

        result = ", ".join(formatted)
        return f"{prefix}{result}" if prefix else result

    def print_summary(self, epoch: int, phase: str = "Valid") -> None:
        averages = self.get_averages()
        print(f"\nEpoch {epoch} [{phase}] Metrics:")
        print(self.format_metrics(averages, prefix="  "))

    def get_metric(self, metric_name: str) -> Optional[float]:
        averages = self.get_averages()
        return averages.get(metric_name)


def perplexity(context):
    loss = context["loss"]
    return float(torch.exp(torch.clamp(torch.tensor(loss), max=50)))


def ppl_to_loss_ratio(outputs, targets, computed_metrics=None):
    if (
        computed_metrics
        and "perplexity" in computed_metrics
        and "loss" in computed_metrics
    ):
        return computed_metrics["perplexity"] / computed_metrics["loss"]
    return None


def accuracy(context: MetricContext) -> float:
    predictions = context.get("predictions")
    targets = context.get("targets")
    assert targets is not None

    if predictions is None:
        # if predictions not pre-computed, get from outputs
        outputs = context["outputs"]
        predictions = torch.argmax(outputs, dim=1)

    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def precision(context: MetricContext) -> float:
    """macro-aveeraged"""

    predictions = context.get("predictions")
    targets = context.get("targets")
    print(f"Total Samples: {targets.size(0)}")
    print(f"Predicted as Positive (1): {(predictions == 1).sum().item()}")
    print(f"Actually Positive (1): {(targets == 1).sum().item()}")

    if predictions is None:
        outputs = context["outputs"]
        predictions = torch.argmax(outputs, dim=1)

    classes = torch.unique(targets)
    precisions = []

    for c in classes:
        tp = ((predictions == c) & (targets == c)).sum().item()
        pred_pos = (predictions == c).sum().item()

        if pred_pos > 0:
            precisions.append(tp / pred_pos)

    return float(sum(precisions) / len(precisions)) if precisions else 0.0


def recall(context: MetricContext) -> float:
    """macro-aveeraged"""
    predictions = context.get("predictions")
    targets = context.get("targets")

    if predictions is None:
        outputs = context["outputs"]
        predictions = torch.argmax(outputs, dim=1)

    classes = torch.unique(targets)
    recalls = []

    for c in classes:
        tp = ((predictions == c) & (targets == c)).sum().item()
        actual_pos = (targets == c).sum().item()

        if actual_pos > 0:
            recalls.append(tp / actual_pos)

    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def f1_score(context) -> float:
    prec = precision(context)
    rec = recall(context)

    if prec + rec == 0:
        return 0.0

    return 2 * (prec * rec) / (prec + rec)


def balanced_accuracy(context) -> float:
    """average of per-class accuracies"""
    predictions = context.get("predictions")
    targets = context.get("targets")

    if predictions is None:
        outputs = context["outputs"]
        predictions = torch.argmax(outputs, dim=1)

    neg_correct = ((predictions == 0) & (targets == 0)).sum().item()
    neg_total = (targets == 0).sum().item()
    neg_acc = neg_correct / neg_total if neg_total > 0 else 0.0

    pos_correct = ((predictions == 1) & (targets == 1)).sum().item()
    pos_total = (targets == 1).sum().item()
    pos_acc = pos_correct / pos_total if pos_total > 0 else 0.0

    return (neg_acc + pos_acc) / 2

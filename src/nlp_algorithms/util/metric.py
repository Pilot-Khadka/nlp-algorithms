from typing import Dict, List, Optional, Any, TypedDict


import torch
from torch import Tensor

from collections import defaultdict


class MetricContext(TypedDict):
    loss: float
    outputs: Tensor
    predictions: Tensor
    targets: Tensor
    tokens: int


class MetricsTracker:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()

    def reset(self) -> None:
        self.batch_metrics: List[Dict[str, float]] = []
        self.accumulated = defaultdict(float)
        self.token_accumulated = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """
        Expected metrics dict may contain:
            loss   -> mean loss for this batch
            tokens -> number of tokens used for this loss
            anything else -> batch-mean metrics
        """

        self.batch_metrics.append(metrics)

        if "loss" in metrics:
            if "tokens" not in metrics:
                raise ValueError(
                    "MetricsTracker.update requires 'tokens' when 'loss' is provided."
                )

            loss = metrics["loss"]
            tokens = metrics["tokens"]

            if isinstance(loss, Tensor):
                loss = loss.item()

            self.accumulated["loss"] += loss * tokens
            # pyrefly: ignore [bad-assignment]
            self.token_accumulated += tokens

        for name, value in metrics.items():
            if name in ("loss", "tokens"):
                continue

            if isinstance(value, Tensor):
                value = value.item()

            self.accumulated[name] += value
            self.accumulated[name + "_count"] += 1

    def get_averages(self) -> Dict[str, float]:
        averages = {}

        if self.token_accumulated > 0:
            averages["loss"] = self.accumulated["loss"] / self.token_accumulated

        for key in list(self.accumulated.keys()):
            if key.endswith("_count"):
                name = key.replace("_count", "")
                count = self.accumulated[key]
                if count > 0:
                    averages[name] = self.accumulated[name] / count

        return averages

    def get_current_batch_metrics(self) -> Dict[str, float]:
        return self.batch_metrics[-1] if self.batch_metrics else {}

    def format_metrics(
        self, metrics: Optional[Dict[str, float]] = None, prefix: str = ""
    ) -> str:
        if metrics is None:
            metrics = self.get_averages()

        if not metrics:
            return f"{prefix}No metrics available"

        formatted = [f"{name}\t{value:.4f}" for name, value in sorted(metrics.items())]
        result = "\t".join(formatted)
        return f"{prefix}{result}" if prefix else result

    def print_summary(self, epoch: int, phase: str = "Valid") -> None:
        averages = self.get_averages()
        print("\n" + "-" * 70)
        print(f"Epoch {epoch} [{phase}] Metrics")
        print("-" * 70)
        print(self.format_metrics(averages))

    def get_metric(self, metric_name: str) -> Optional[float]:
        return self.get_averages().get(metric_name)


def perplexity(context):
    loss = context["loss"]
    return float(torch.exp(torch.tensor(loss)))


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

    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total if total > 0 else 0.0


def precision(context: MetricContext) -> float:
    """description: macro-aveeraged"""

    predictions = context.get("predictions")
    targets = context.get("targets")

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

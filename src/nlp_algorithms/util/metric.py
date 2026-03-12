from typing import TypedDict

import math
import torch
from torch import Tensor


class MetricContext(TypedDict):
    loss: float
    outputs: Tensor
    predictions: Tensor
    targets: Tensor
    tokens: int


def perplexity(loss):
    if isinstance(loss, torch.Tensor):
        loss = loss.detach().item()
    return math.exp(loss)


def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro precision across all classes."""
    precisions = []
    for c in range(num_classes):
        tp = ((predictions == c) & (targets == c)).sum().item()
        pred_pos = (predictions == c).sum().item()
        prec = tp / pred_pos if pred_pos > 0 else 0.0
        precisions.append(prec)

    return sum(precisions) / num_classes if num_classes > 0 else 0.0


def recall(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    """Macro recall across all classes."""
    recalls = []
    for c in range(num_classes):
        tp = ((predictions == c) & (targets == c)).sum().item()
        actual_pos = (targets == c).sum().item()
        rec = tp / actual_pos if actual_pos > 0 else 0.0
        recalls.append(rec)

    return sum(recalls) / num_classes if num_classes > 0 else 0.0


def f1_score(
    predictions: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> float:
    """Macro F1 (computed from macro P/R)."""
    prec = precision(predictions, targets, num_classes)
    rec = recall(predictions, targets, num_classes)

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

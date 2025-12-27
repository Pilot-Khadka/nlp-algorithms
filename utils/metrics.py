from typing import Dict, List, Optional


import math
import torch
from collections import defaultdict


class MetricsTracker:
    def __init__(self, metric_names: List[str]):
        self.metric_names = metric_names
        self.reset()

    def reset(self) -> None:
        self.batch_metrics: List[Dict[str, float]] = []
        self.accumulated: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)

    def update(self, batch_metrics: Dict[str, float], batch_size: int = 1) -> None:
        self.batch_metrics.append(batch_metrics)

        for metric_name, value in batch_metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.accumulated[metric_name] += value * batch_size
            self.counts[metric_name] += batch_size

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

        formatted = []
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


def loss(outputs, targets):
    return torch.nn.functional.cross_entropy(
        outputs.view(-1, outputs.size(-1)), targets.view(-1)
    ).item()


def perplexity(outputs, targets, computed_metrics=None):
    if computed_metrics and "loss" in computed_metrics:
        return math.exp(computed_metrics["loss"])
    return math.exp(loss(outputs, targets))


def ppl_to_loss_ratio(outputs, targets, computed_metrics=None):
    if (
        computed_metrics
        and "perplexity" in computed_metrics
        and "loss" in computed_metrics
    ):
        return computed_metrics["perplexity"] / computed_metrics["loss"]
    return None

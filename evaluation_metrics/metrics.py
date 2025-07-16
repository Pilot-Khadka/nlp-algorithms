import math

# global registry
METRICS_REGISTERY = {}


def register_metric(name):
    def decorator(fn):
        METRICS_REGISTERY[name] = fn
        return fn

    return decorator


def compute_metrics(names, **kwargs):
    """
    names: list of metric names to compute, e.g. ["perplexity","accuracy"]
    kwargs: whatever each metric needs (avg_loss, predictions, targets...)
    """
    results = {}
    for name in names:
        if name not in METRICS_REGISTERY:
            raise KeyError(f"Metric '{name}' is not registered.")
        fn = METRICS_REGISTERY[name]
        results.update(fn(**kwargs))
    return results


@register_metric("perplexity")
def perplexity(avg_loss=None, **kwargs):
    if avg_loss is None:
        raise ValueError("perplexity needs avg loss")
    return {"perplexity": math.exp(avg_loss)}


@register_metric("accuracy")
def accuracy(predictions=None, targets=None, **kwargs):
    if predictions is None or targets is None:
        raise ValueError("accuracy needs predictionns and targets ")
    preds = predictions.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return {"accuracy": correct / total}

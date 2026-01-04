from typing import Union, Optional, Dict, Any

import yaml
from pathlib import Path

import torch
import torch.nn as nn


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key)
        return value

    def __setattr__(self, key, value):
        self[key] = value


def convert_numeric(value):
    if isinstance(value, str):
        if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
            return int(value)
        try:
            float_val = float(value)
            return float_val
        except ValueError:
            return value
    return value


def to_attrdict(obj):
    if isinstance(obj, dict):
        return AttrDict({k: to_attrdict(convert_numeric(v)) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [to_attrdict(convert_numeric(v)) for v in obj]
    else:
        return convert_numeric(obj)


def load_config(path: str):
    raw_cfg = load_yaml(path)
    print("raw cfg:", raw_cfg)

    if not isinstance(raw_cfg, dict):
        raise ValueError("Top-level YAML must be a mapping")

    return raw_cfg


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    epoch: int,
    lr: float,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    metric_value: float,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "lr": lr,
        "optimizer": optimizer.state_dict(),
        "model": model.state_dict(),
        "metric_value": metric_value,
    }

    if additional_info:
        checkpoint.update(additional_info)

    torch.save(checkpoint, checkpoint_path)
    print("Model saved at:", checkpoint_path)


def load_model(filepath, model_class, embedding_class=nn.Embedding, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)

    model = model_class(**checkpoint["model_kwargs"])
    embedding = embedding_class(
        checkpoint["model_kwargs"]["vocab_size"],
        checkpoint["model_kwargs"]["embedding_dim"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    embedding.load_state_dict(checkpoint["embedding_state_dict"])

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding.parameters())
    )
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "model": model,
        "embedding": embedding,
        "optimizer": optimizer,
        "epoch": checkpoint["epoch"],
        "loss": checkpoint["loss"],
        "model_kwargs": checkpoint["model_kwargs"],
    }

from typing import Union, Optional, Dict, Any


import os
import sys
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn


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

from typing import Union, Optional, Dict, Any

import yaml
from pathlib import Path

import torch
import torch.nn as nn


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_num_workers(num_workers) -> int:
    import multiprocessing

    if isinstance(num_workers, int) and num_workers > 0:
        return num_workers

    num_cpus = multiprocessing.cpu_count()
    return min(8, max(0, num_cpus - 2))


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


def convert_to_attrdict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return AttrDict(
            {k: convert_to_attrdict(convert_numeric(v)) for k, v in obj.items()}
        )
    elif isinstance(obj, list):
        return [convert_to_attrdict(convert_numeric(v)) for v in obj]
    else:
        return convert_numeric(obj)


def load_config(path: str) -> AttrDict:
    raw_cfg = load_yaml(path)
    print(yaml.safe_dump(raw_cfg, sort_keys=False, default_flow_style=False))

    if not isinstance(raw_cfg, dict):
        raise ValueError("Top-level YAML must be a mapping")

    cfg = convert_to_attrdict(raw_cfg)
    assert isinstance(cfg, AttrDict)
    return cfg


def save_checkpoint(
    checkpoint_path: Union[str, Path],
    epoch: int,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    scheduler,
    best_valid_loss: Optional[float] = None,
    additional_info: Optional[Dict[str, Any]] = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_valid_loss": best_valid_loss,
        "rng_state": {
            "torch": torch.get_rng_state().clone(),
            "cuda": (
                [s.clone() for s in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else None
            ),
        },
    }

    if scheduler is not None:
        checkpoint["scheduler_state"] = scheduler.state_dict()

    if additional_info is not None:
        checkpoint["additional_info"] = additional_info

    torch.save(checkpoint, checkpoint_path)
    # print("Model saved at:", checkpoint_path)


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


def load_checkpoint(
    checkpoint_path,
    model,
    optimizer,
    scheduler=None,
    device=torch.device("cpu"),
):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    if "rng_state" in checkpoint:
        torch_state = checkpoint["rng_state"].get("torch")
        if torch_state is not None:
            torch.set_rng_state(torch_state.detach().cpu().to(torch.uint8))

        cuda_state = checkpoint["rng_state"].get("cuda")
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(
                [s.detach().cpu().to(torch.uint8) for s in cuda_state]
            )
    start_epoch = checkpoint["epoch"]
    best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))

    return start_epoch, best_valid_loss

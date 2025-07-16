import os
import torch
import logging
import torch.nn as nn


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "training.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def save_model(model, embedding, optimizer, epoch, loss, filepath, **model_kwargs):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "embedding_state_dict": embedding.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "model_kwargs": model_kwargs,
        "vocab_size": model_kwargs.get("output_dim"),
    }
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)


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

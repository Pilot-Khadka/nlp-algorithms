import os
import sys
import importlib.util
import torch
import logging
import torch.nn as nn


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # prevent adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # decide where logs go
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # for real time feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


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


def import_function_from_folder(
    folder_path: str, module_filename: str, function_name: str
):
    """
    Because the folders are named starting from number and a .
    normal import would not work
    """
    module_path = os.path.abspath(os.path.join(folder_path, module_filename))
    module_name = f"dynamic_module_{os.path.basename(folder_path)}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        raise ImportError(f"Cannot find module spec for {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    func = getattr(module, function_name, None)
    if func is None:
        raise AttributeError(
            f"""
            Function '{function_name}' not found in module '{module_filename}'
            """
        )

    return func

import sys
import torch
from tqdm import tqdm
from utils.metrics import compute_metrics


def train(
    model,
    task,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device,
    logger,
    config,
):
    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)

        for batch_idx, batch in enumerate(train_loader):
            loss = task.train_step(batch, model, criterion, optimizer, device)
            total_loss += loss
            # progress bar using sys.stdout
            progress = (batch_idx + 1) / num_batches
            bar_width = 30
            bar = "#" * int(progress * bar_width) + "-" * (
                bar_width - int(progress * bar_width)
            )
            sys.stdout.write(
                f"\rEpoch {epoch + 1} [{bar}] {batch_idx + 1}/{num_batches} "
                f"Loss: {loss:.4f}"
            )
            sys.stdout.flush()
        avg_loss = total_loss / num_batches
        sys.stdout.write("\n")
        logger.info(f"Epoch {epoch + 1}: Train loss: {avg_loss:.4f}")
        logger.info(
            f"Epoch {epoch + 1}: Train loss: {total_loss / len(train_loader)}")


def train_epoch(
    model,
    embedding,
    train_loader,
    optimizer,
    criterion,
    device,
    logger,
    metrics_fn=None,
):
    model.train()
    embedding.train()

    total_loss = 0
    num_batches = len(train_loader)

    with tqdm(train_loader, desc="Training", leave=False) as pbar:
        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)

            x_embed = embedding(x)
            output = model(x_embed)
            loss = criterion(output, y[:, -1])  # Predict last token

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"}
            )

    avg_loss = total_loss / num_batches
    metrics = {}
    if metrics_fn is not None:
        metrics = compute_metrics(metrics_fn, avg_loss=avg_loss)

    for k, v in metrics.items():
        logger.info(f"Training - {k}: {v:.4f}")
    logger.info(f"Training - Average Loss: {avg_loss:.4f}")
    return avg_loss, metrics


def validate_epoch(
    model, embedding, valid_loader, criterion, device, logger, metrics_fn=None
):
    model.eval()
    embedding.eval()

    total_loss = 0
    num_batches = len(valid_loader)

    with torch.no_grad():
        with tqdm(valid_loader, desc="Validation", leave=False) as pbar:
            for batch_idx, (x, y) in enumerate(pbar):
                x, y = x.to(device), y.to(device)

                x_embed = embedding(x)
                output = model(x_embed)
                loss = criterion(output, y[:, -1])  # Predict last token

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)

                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "avg_loss": f"{avg_loss:.4f}"}
                )

    avg_loss = total_loss / num_batches
    metrics = {}
    if metrics_fn is not None:
        metrics = compute_metrics(metrics_fn, avg_loss=avg_loss)

    for k, v in metrics.items():
        logger.info(f"Validation - {k}: {v:.4f}")
    logger.info(f"Validation - Average Loss: {avg_loss:.4f}")
    return avg_loss, metrics

import torch
from tqdm import tqdm
from evaluation_metrics.metrics import compute_metrics


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

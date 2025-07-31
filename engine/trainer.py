import torch
from tqdm import tqdm


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
        total_train_loss = 0
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']} [Train]",
            leave=False,
        )

        for batch_idx, batch in enumerate(train_progress):
            loss = task.train_step(batch, model, criterion, optimizer, device)
            total_train_loss += loss

            train_progress.set_postfix(
                {
                    "Loss": f"{loss:.4f}",
                    "Avg Loss": f"{total_train_loss / (batch_idx + 1):.4f}",
                }
            )
        avg_train_loss = total_train_loss / len(train_loader)
        model.eval()
        total_valid_loss = 0
        all_metrics = []
        valid_progress = tqdm(
            valid_loader,
            desc=f"Epoch {epoch + 1}/{config['epochs']} [Valid]",
            leave=False,
        )

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_progress):
                loss, batch_metrics = task.eval_step(batch, model, criterion, device)
                total_valid_loss += loss
                all_metrics.append(batch_metrics)

                valid_progress.set_postfix(
                    {
                        "Val Loss": f"{loss:.4f}",
                        "Avg Val Loss": f"{total_valid_loss / (batch_idx + 1):.4f}",
                    }
                )

        avg_valid_loss = total_valid_loss / len(valid_loader)
        aggregated_metrics = aggregate_metrics(all_metrics)

        logger.info(
            f"""Epoch {epoch + 1}: Train loss: {avg_train_loss:.4f}, Valid loss: {
                avg_valid_loss:.4f}"""
        )

        display_metrics_matrix(logger, aggregated_metrics)


def display_metrics_matrix(logger, metrics):
    if not metrics:
        return

    filtered_metrics = {k: v for k, v in metrics.items() if k != "batch_size"}

    per_class_metrics = {}
    overall_metrics = {}

    for key, value in filtered_metrics.items():
        if "_class_" in key:
            parts = key.split("_class_")
            metric_type = parts[0]
            class_idx = int(parts[1])

            if metric_type not in per_class_metrics:
                per_class_metrics[metric_type] = {}
            per_class_metrics[metric_type][class_idx] = value
        else:
            overall_metrics[key] = value

    logger.info("Valid Metrics:")

    if overall_metrics:
        for metric_name, metric_value in overall_metrics.items():
            logger.info(f"  {metric_name}: {metric_value:.4f}")

    if per_class_metrics:
        metric_types = sorted(per_class_metrics.keys())
        all_classes = set()
        for metric_dict in per_class_metrics.values():
            all_classes.update(metric_dict.keys())
        all_classes = sorted(all_classes)

        if metric_types and all_classes:
            logger.info("  Per-Class:")

            header = "    Class"
            separator = "    -----"
            for metric_type in metric_types:
                header += f" | {metric_type:>9}"
                separator += f"-|-{'-' * 9}"

            logger.info(header)
            logger.info(separator)

            for class_idx in all_classes:
                row = f"    {class_idx:5d}"
                for metric_type in metric_types:
                    value = per_class_metrics[metric_type].get(class_idx, 0.0)
                    row += f" | {value:9.4f}"
                logger.info(row)


def aggregate_metrics(all_metrics):
    if not all_metrics:
        return {}

    metric_names = [key for key in all_metrics[0].keys() if key != "batch_size"]

    aggregated = {}
    total_samples = sum(metrics["batch_size"] for metrics in all_metrics)

    for metric_name in metric_names:
        weighted_sum = sum(
            metrics[metric_name] * metrics["batch_size"] for metrics in all_metrics
        )
        aggregated[metric_name] = weighted_sum / total_samples

    return aggregated

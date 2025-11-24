import os
import sys
import hydra
import torch
import questionary
from omegaconf import DictConfig, OmegaConf

from engine.task_factory import load_task
from engine.model_factory import ModelFactory
from datasets.loader import load_dataset
from engine.trainer import train
from utils.logger import setup_logging


def get_available_choices(config_path):
    path = os.path.join("conf", config_path)
    if not os.path.isdir(path):
        print(f"Error: Configuration directory not found at {path}", file=sys.stderr)
        return []

    choices = [
        f.rsplit(".", 1)[0]
        for f in os.listdir(path)
        if f.endswith((".yaml", ".yml"))
        and f
        not in (
            "config.yaml",
            "placeholder_model.yaml",
            "placeholder_task.yaml",
            "placeholder_dataset.yaml",
        )
    ]
    return sorted(choices)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print("\n---  Configuration Setup ---\n")

    available_tasks = get_available_choices("task")
    if not available_tasks:
        print("Error: No tasks found. Check 'conf/task' directory.", file=sys.stderr)
        return

    task_name = questionary.select(
        "Which **Task** would you like to run?",
        choices=available_tasks,
    ).ask()

    if task_name is None:
        return

    task_cfg_path = os.path.join("conf", "task", f"{task_name}.yaml")
    if not os.path.exists(task_cfg_path):
        print(f"Error: Task config file not found at {task_cfg_path}", file=sys.stderr)
        return

    selected_task_cfg = OmegaConf.load(task_cfg_path)
    compatible_models = selected_task_cfg.get("compatible_models", [])
    available_models = get_available_choices("model")
    model_choices = [m for m in available_models if m in compatible_models]

    if not model_choices:
        print(
            f"Error: No compatible models found for task '{task_name}'.",
            file=sys.stderr,
        )
        return

    model_name = questionary.select(
        f"Choose a **Model** (Compatible with {task_name}):",
        choices=model_choices,
    ).ask()

    if model_name is None:
        return

    available_datasets = get_available_choices("dataset")
    compatible_datasets = selected_task_cfg.get("compatible_datasets", [])
    dataset_choices = [m for m in available_datasets if m in compatible_datasets]
    if not dataset_choices:
        print(
            "Error: No datasets found. Check 'conf/dataset' directory.", file=sys.stderr
        )
        return

    dataset_name = questionary.select(
        "Choose a **Dataset**:",
        choices=dataset_choices,
    ).ask()

    if dataset_name is None:
        return

    cli_overrides = [
        f"task={task_name}",
        f"model={model_name}",
        f"dataset={dataset_name}",
    ]

    cfg_resolved = hydra.compose(config_name="config", overrides=cli_overrides)
    cfg = cfg_resolved

    print(f"\n **Running with configuration:**\n{OmegaConf.to_yaml(cfg)}")

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_bundle = load_dataset(cfg)
    task = load_task(cfg.task.name)

    factory = ModelFactory()

    model = factory.create_model(
        cfg.model,
        dataset_bundle,
        task,
    )
    model.to(device)

    criterion = task.get_loss_function()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    metrics_to_use = cfg.task.get("metrics", [])
    logger.info(f"Task metrics: {metrics_to_use}")

    train_loader = dataset_bundle.train_loader
    valid_loader = dataset_bundle.valid_loader
    train(
        model,
        task,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        device,
        logger,
        cfg.training,
    )


if __name__ == "__main__":
    main()

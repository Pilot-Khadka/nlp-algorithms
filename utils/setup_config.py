import os
import sys
import questionary
from omegaconf import OmegaConf
from typing import List, Optional


def get_available_choices(config_path: str) -> List[str]:
    path = os.path.join("conf", config_path)
    if not os.path.isdir(path):
        print(f"Error: Configuration directory not found at {path}", file=sys.stderr)
        return []

    excluded_files = {
        "config.yaml",
        "placeholder_model.yaml",
        "placeholder_task.yaml",
        "placeholder_dataset.yaml",
        "default.yaml",
    }

    choices = [
        f.rsplit(".", 1)[0]
        for f in os.listdir(path)
        if f.endswith((".yaml", ".yml")) and f not in excluded_files
    ]
    return sorted(choices)


def load_task_config(task_name: str) -> Optional[OmegaConf]:
    task_cfg_path = os.path.join("conf", "task", f"{task_name}.yaml")
    if not os.path.exists(task_cfg_path):
        print(f"Error: Task config file not found at {task_cfg_path}", file=sys.stderr)
        return None
    return OmegaConf.load(task_cfg_path)


def prompt_task_selection() -> Optional[str]:
    available_tasks = get_available_choices("task")
    if not available_tasks:
        print("Error: No tasks found. Check 'conf/task' directory.", file=sys.stderr)
        return None

    return questionary.select(
        "Which **Task** would you like to run?",
        choices=available_tasks,
    ).ask()


def prompt_model_selection(task_name: str, task_cfg: OmegaConf) -> Optional[str]:
    compatible_models = task_cfg.get("compatible_models", [])
    available_models = get_available_choices("model")
    model_choices = [m for m in available_models if m in compatible_models]

    if not model_choices:
        print(
            f"Error: No compatible models found for task '{task_name}'.",
            file=sys.stderr,
        )
        return None

    return questionary.select(
        f"Choose a **Model** (Compatible with {task_name}):",
        choices=model_choices,
    ).ask()


def prompt_dataset_selection(task_cfg: OmegaConf) -> Optional[str]:
    available_datasets = get_available_choices("dataset")
    compatible_datasets = task_cfg.get("compatible_datasets", [])
    dataset_choices = [d for d in available_datasets if d in compatible_datasets]

    if not dataset_choices:
        print(
            "Error: No datasets found. Check 'conf/dataset' directory.",
            file=sys.stderr,
        )
        return None

    return questionary.select(
        "Choose a **Dataset**:",
        choices=dataset_choices,
    ).ask()


def setup_configuration() -> Optional[tuple]:
    print("\n--- Configuration Setup ---\n")

    task_name = prompt_task_selection()
    if task_name is None:
        return None

    task_cfg = load_task_config(task_name)
    if task_cfg is None:
        return None

    model_name = prompt_model_selection(task_name, task_cfg)
    if model_name is None:
        return None

    dataset_name = prompt_dataset_selection(task_cfg)
    if dataset_name is None:
        return None

    return task_name, model_name, dataset_name

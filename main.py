import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from engine.trainer import train
from utils.logger import setup_logging
from datasets.loader import load_dataset
from engine.task_factory import load_task
from engine.model_factory import ModelFactory
from utils.setup_config import setup_configuration


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    config_result = setup_configuration()
    if config_result is None:
        return

    task_name, model_name, dataset_name = config_result

    cli_overrides = [
        f"task={task_name}",
        f"model={model_name}",
        f"dataset={dataset_name}",
        f"training={model_name}",
    ]

    cfg_resolved = hydra.compose(config_name="config", overrides=cli_overrides)

    print(f"\n**Running with configuration:**\n{OmegaConf.to_yaml(cfg_resolved)}")

    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_bundle = load_dataset(cfg_resolved)

    task = load_task(cfg_resolved.task.name)

    factory = ModelFactory()
    model = factory.create_model(
        cfg_resolved.model,
        dataset_bundle,
        task,
    )
    model.to(device)

    criterion = task.get_loss_function()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg_resolved.training.learning_rate, weight_decay=1e-4
    )

    metrics_to_use = cfg_resolved.task.get("metrics", [])
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
        cfg_resolved.training,
    )


if __name__ == "__main__":
    main()

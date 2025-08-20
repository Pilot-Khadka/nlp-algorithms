import hydra
import torch
from omegaconf import DictConfig

from engine.task_factory import load_task
from engine.model_factory import ModelFactory
from datasets.loader import load_dataset
from engine.trainer import train
from utils.logger import setup_logging


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_bundle = load_dataset(cfg)
    task = load_task(cfg.task.name)

    factory = ModelFactory()
    print("Available models:", factory.list_available_models())

    model = factory.create_model(
        cfg.model,
        dataset_bundle,
        task,
    )
    print("mode:", model)
    model.to(device)

    criterion = task.get_loss_function()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

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

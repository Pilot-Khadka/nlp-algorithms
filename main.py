import hydra
import torch
from omegaconf import DictConfig
import torch.nn as nn

from engine.task_factory import load_task
from engine.model_factory import create_model
from datasets.loader import load_dataset
from engine.trainer import train
from utils.logger import setup_logging
from models.model_registry import load_model_from_name, load_vocab


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_bundle = load_dataset(cfg.dataset)
    task = load_task(cfg.task.name)

    if cfg.model.use_pretrained_embedding:
        assert cfg.model.embedding_path is not None, "embedding_path is required"
        assert cfg.model.vocab_path is not None, "vocab_path is required"

        w2v_model = load_model_from_name(cfg.model.embedding_path)
        vocab = load_vocab(cfg.model.vocab_path)
        pretrained_weights = w2v_model.get_input_embeddings().clone().detach().cpu()
        embedding_layer = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
    else:
        embedding_layer = None
        word2idx = None

    model = create_model(
        cfg.model, dataset_bundle, task, embedding_layer=embedding_layer
    )

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

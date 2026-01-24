from engine.registry import get_from_registry, TRAINER_REGISTRY


def main():
    # from train_util import run_training
    from util.util import load_config

    cfg = load_config(path="config/bigru_imdb.yaml")
    # run_training(cfg_resolved=cfg)
    trainer = get_from_registry(registry=TRAINER_REGISTRY, name=cfg.task.name)
    trainer(config=cfg).train()


if __name__ == "__main__":
    main()

from util.util import load_config
from training.train_orchestrator import run_training


def main():
    config = load_config(path="config/bigru_imdb.yaml")
    run_training(config)


if __name__ == "__main__":
    main()

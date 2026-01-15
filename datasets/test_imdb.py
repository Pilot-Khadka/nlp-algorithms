from utils.utils import load_config, to_attrdict
from datasets.imdb import get_imdb_dataloaders


def main():
    cfg = load_config(path=".../config/lstm_imdb_simple.yaml")

    dataset_bundle = get_imdb_dataloaders(cfg)


if __name__ == "__main__":
    main()

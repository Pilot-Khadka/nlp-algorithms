from dataset.tatoeba import get_tatoeba_dataloaders
from util.util import load_config


if __name__ == "__main__":
    cfg = load_config(path="../config/seq2seq_tatoeba.yaml")
    dataloaders = get_tatoeba_dataloaders(cfg)

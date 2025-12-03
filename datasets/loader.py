from datasets.imdb import get_imdb_dataloaders
from datasets.penn_treebank import get_ptb_dataloaders
from datasets.sst2 import get_sst2_dataloaders
from datasets.tatoeba_dataset import get_tatoeba_dataloaders


def load_dataset(cfg):
    if cfg.dataset.name == "ptb":
        return get_ptb_dataloaders(cfg)
    elif cfg.dataset.name == "sst2":
        return get_sst2_dataloaders(cfg)
    elif cfg.dataset.name == "imdb":
        return get_imdb_dataloaders(cfg)
    elif cfg.dataset.name == "tatoeba":
        return get_tatoeba_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

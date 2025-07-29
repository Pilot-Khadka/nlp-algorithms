from datasets.penn_treebank.penn_treebank import get_ptb_dataloaders
from datasets.sst2.sst2 import get_sst2_dataloaders


def load_dataset(cfg_dataset):
    if cfg_dataset.name == "ptb":
        return get_ptb_dataloaders(cfg_dataset)
    elif cfg_dataset.name == "sst2":
        return get_sst2_dataloaders(cfg_dataset)
    else:
        raise ValueError(f"Unknown dataset name: {cfg_dataset.name}")

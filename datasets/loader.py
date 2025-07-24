from datasets.penn_treebank import get_ptb_dataloaders


def load_dataset(cfg_dataset):
    if cfg_dataset.name == "ptb":
        return get_ptb_dataloaders(cfg_dataset)
    else:
        raise ValueError(f"Unknown dataset name: {cfg_dataset.name}")

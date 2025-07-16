from datasets.penn_treebank import get_ptb_dataloaders


def load_dataset(name, **kwargs):
    if name == "ptb":
        return get_ptb_dataloaders(**kwargs)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

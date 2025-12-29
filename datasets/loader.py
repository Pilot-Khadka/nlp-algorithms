def load_dataset(cfg):
    if cfg.datasets.name == "ptb":
        from datasets.penn_treebank import get_ptb_dataloaders

        return get_ptb_dataloaders(cfg)
    elif cfg.datasets.name == "sst2":
        from datasets.sst2 import get_sst2_dataloaders

        return get_sst2_dataloaders(cfg)
    elif cfg.datasets.name == "imdb":
        from datasets.imdb import get_imdb_dataloaders

        return get_imdb_dataloaders(cfg)
    elif cfg.datasets.name == "tatoeba":
        from datasets.tatoeba import get_tatoeba_dataloaders

        return get_tatoeba_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.datasets.name}")


def ensure_dataset_exists(cfg):
    if cfg.datasets.name == "ptb":
        import os
        from datasets.penn_treebank import PTBDataset

        data_dir = cfg.datasets["data_dir"]

        if not os.path.exists(data_dir):
            ds = PTBDataset(cfg, split="train")
            del ds

    elif cfg.datasets.name == "sst2":
        from datasets.sst2 import get_sst2_dataloaders

        return get_sst2_dataloaders(cfg)
    elif cfg.datasets.name == "imdb":
        from datasets.imdb import get_imdb_dataloaders

        return get_imdb_dataloaders(cfg)
    elif cfg.datasets.name == "tatoeba":
        from datasets.tatoeba import get_tatoeba_dataloaders

        return get_tatoeba_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.datasets.name}")

def load_dataset(cfg):
    if cfg.dataset.name == "ptb":
        from dataset.penn_treebank import get_ptb_dataloaders

        return get_ptb_dataloaders(cfg)
    elif cfg.dataset.name == "sst2":
        from dataset.sst2 import get_sst2_dataloaders

        return get_sst2_dataloaders(cfg)
    elif cfg.dataset.name == "imdb":
        from dataset.imdb import get_imdb_dataloaders

        return get_imdb_dataloaders(cfg)
    elif cfg.dataset.name == "tatoeba":
        from dataset.tatoeba import get_tatoeba_dataloaders

        return get_tatoeba_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")


def ensure_dataset_exists(cfg):
    if cfg.dataset.name == "ptb":
        import os
        from dataset.penn_treebank import PTBCorpus

        data_dir = cfg.dataset["data_dir"]

        if not os.path.exists(data_dir):
            ds = PTBCorpus(cfg, split="train")
            del ds

    elif cfg.dataset.name == "sst2":
        from dataset.sst2 import get_sst2_dataloaders

        return get_sst2_dataloaders(cfg)
    elif cfg.dataset.name == "imdb":
        from dataset.imdb import get_imdb_dataloaders

        return get_imdb_dataloaders(cfg)
    elif cfg.dataset.name == "tatoeba":
        from dataset.tatoeba import get_tatoeba_dataloaders

        return get_tatoeba_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

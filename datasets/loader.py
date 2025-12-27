def load_dataset(cfg):
    if cfg.dataset.name == "ptb":
        from datasets.penn_treebank import get_ptb_dataloaders

        return get_ptb_dataloaders(cfg)
    elif cfg.dataset.name == "sst2":
        from datasets.sst2 import get_sst2_dataloaders

        return get_sst2_dataloaders(cfg)
    elif cfg.dataset.name == "imdb":
        from datasets.imdb import get_imdb_dataloaders

        return get_imdb_dataloaders(cfg)
    elif cfg.dataset.name == "tatoeba":
        from datasets.tatoeba import get_tatoeba_dataloaders

        return get_tatoeba_dataloaders(cfg)
    else:
        raise ValueError(f"Unknown dataset name: {cfg.dataset.name}")

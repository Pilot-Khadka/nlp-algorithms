from hydra import initialize, compose
from main import main

with initialize(config_path="conf", version_base=None):
    cfg = compose(
        config_name="config",
        overrides=[
            "task=language_modeling",
            "model=lstm",
            "dataset=ptb",
            "training=lstm",
        ],
    )
    main(cfg)

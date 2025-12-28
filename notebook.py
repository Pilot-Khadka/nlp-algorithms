import sys
from hydra import initialize, compose
from main import main

if __name__ == "__main__":
    if not hasattr(sys.modules["__main__"], "__spec__"):
        sys.modules["__main__"].__spec__ = None

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

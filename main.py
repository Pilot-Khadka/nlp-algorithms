import os
from omegaconf import OmegaConf

from train_utils import run_training


def load_config():
    cfg = OmegaConf.load("config/config.yaml")

    experiment_name = cfg.defaults[0]["experiment"]
    experiment_cfg = OmegaConf.load(
        os.path.join("config", "experiments", f"{experiment_name}.yaml")
    )

    composed = [cfg]
    for entry in experiment_cfg.get("defaults", []):
        for group, name in entry.items():
            path = os.path.join("config", group, f"{name}.yaml")
            sub_cfg = OmegaConf.load(path)
            grouped_cfg = OmegaConf.create({group: sub_cfg})
            composed.append(grouped_cfg)

    full_cfg = OmegaConf.merge(*composed)
    full_cfg.pop("defaults", None)
    return full_cfg


def main():
    cfg = load_config()
    print(OmegaConf.to_yaml(cfg))
    run_training(cfg)


if __name__ == "__main__":
    main()

import yaml
from train_utils import run_training

from utils.utils import load_config, to_attrdict


def main():
    cfg = load_config(path="config/lstm_ptb.yaml")
    print("Loaded YAML:", yaml.dump(cfg, sort_keys=False))

    cfg_resolved = to_attrdict(cfg)
    run_training(cfg_resolved=cfg_resolved)


if __name__ == "__main__":
    main()

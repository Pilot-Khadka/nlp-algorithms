def main():
    from train_util import run_training
    from util.util import load_config

    cfg = load_config(path="config/bilstm_imdb.yaml")

    run_training(cfg_resolved=cfg)


if __name__ == "__main__":
    main()

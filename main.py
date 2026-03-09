def main():
    from nlp_algorithms.util import load_config
    from nlp_algorithms.training import run_training

    config = load_config(path="config/bilstm_imdb.yaml")
    run_training(config)


if __name__ == "__main__":
    main()

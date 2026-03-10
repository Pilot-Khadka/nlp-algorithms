import argparse


def main():
    from nlp_algorithms.util import load_config
    from nlp_algorithms.training import run_training

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the config YAML file."
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="If set, show training and validation progress bars.",
    )
    args = parser.parse_args()

    config = load_config(path=args.path)
    config.show_progress = args.show_progress
    run_training(config)


if __name__ == "__main__":
    main()

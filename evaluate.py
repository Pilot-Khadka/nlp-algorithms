import argparse
from pathlib import Path


def main():
    from nlp_algorithms.util import load_config
    from nlp_algorithms.training import run_evaluation

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the config YAML file."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint to evaluate. Defaults to the one saved by train.py "
        "(model_best.pt, for translation also the average of the last epochs).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If set, show evaluation progress bars.",
    )
    args = parser.parse_args()

    config = load_config(path=args.path)
    config.show_progress = args.verbose
    config.config_name = Path(args.path).stem
    run_evaluation(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()

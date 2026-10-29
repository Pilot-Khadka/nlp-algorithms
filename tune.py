import torch
from nlp_algorithms.encoder_decoder.data import load_vocab, load_ne_en
from nlp_algorithms.encoder_decoder.tune import tune_hyperparameters
from nlp_algorithms.encoder_decoder.train import train_model


def tune_and_train(
    load_fn=load_ne_en,
    vocab_path="ne_en_vocab.pt",
    n_random_trials: int = 30,
    n_bayesian_trials: int = 100,
    data_fraction: float = 0.02,
    num_epochs_per_trial: int = 5,
    num_final_epochs: int = 8,
    output_prefix: str = "ne_en_model_tuned_",
) -> dict:
    n_gpus = torch.cuda.device_count()
    devices = (
        [torch.device(f"cuda:{i}") for i in range(n_gpus)]
        if n_gpus > 0
        else [torch.device("cpu")]
    )

    vocab_src, vocab_tgt = load_vocab(load_fn=load_fn, vocab_path=vocab_path)

    best_params = tune_hyperparameters(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        load_fn=load_fn,
        n_random_trials=n_random_trials,
        n_bayesian_trials=n_bayesian_trials,
        num_epochs_per_trial=num_epochs_per_trial,
        data_fraction=data_fraction,
        devices=devices,
    )
    print("best params:", best_params)

    config = {
        "batch_size": best_params["batch_size"],
        "distributed": n_gpus > 1,
        "num_epochs": num_final_epochs,
        "accum_iter": 10,
        "base_lr": best_params["lr"],
        "max_padding": 128,
        "warmup": best_params["warmup"],
        "num_layers": best_params["num_layers"],
        "d_model": best_params["d_model"],
        "file_prefix": output_prefix,
    }

    print(f"\nLaunching full training with best config: {config}")

    return config


if __name__ == "__main__":
    tune_and_train()

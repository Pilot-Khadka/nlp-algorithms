from os.path import exists

import torch

from nlp_algorithms.encoder_decoder.seq2seq import make_model
from nlp_algorithms.encoder_decoder.train import train_model
from nlp_algorithms.encoder_decoder.data import load_vocab


def load_trained_model(model_path="multi30k_model_final.pt"):
    vocab_src, vocab_tgt = load_vocab()

    config = {
        "batch_size": 32,
        "distributed": False,
        "num_epochs": 8,
        "accum_iter": 10,
        "base_lr": 1.0,
        "max_padding": 72,
        "warmup": 3000,
        "num_layers": 6,
        "d_model": 512,
        "file_prefix": "multi30k_model_",
    }

    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, config)

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


if __name__ == "__main__":
    model = load_trained_model()

from os.path import exists

import torch

from nlp_algorithms.encoder_decoder.seq2seq import make_model
from nlp_algorithms.encoder_decoder.train import train_model
from nlp_algorithms.encoder_decoder.data import load_vocab, load_ne_en, load_multi30k


def load_trained_model(
    load_fn=load_multi30k,
    vocab_path="de_en_vocab.pt",
    model_path="de_en_model_final.pt",
):
    vocab_src, vocab_tgt = load_vocab(load_fn=load_fn, vocab_path=vocab_path)

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
        "file_prefix": "en_ne_model_",
    }

    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, config, load_fn=load_fn)

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

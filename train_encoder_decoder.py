from os.path import exists

import torch

from nlp_algorithms.encoder_decoder.seq2seq import make_model
from nlp_algorithms.encoder_decoder.train import train_model
from nlp_algorithms.encoder_decoder.data import load_vocab, load_ne_en, load_multi30k


def load_trained_model(
    load_fn=load_ne_en,
    vocab_path="checkpoint/ne_en/vocab.pt",
    model_path="checkpoint/ne_en/model_best.pt",
):
    # joint bpe so the source, target and output embeddings can be shared
    vocab_src, vocab_tgt = load_vocab(
        load_fn=load_fn,
        vocab_path=vocab_path,
        vocab_size=4000,
        joint=True,
        sample_size=60000,
    )

    # close to duwal & bal 2019 (fairseq, 13.0 bleu ne->en on the same 208k pairs)
    config = {
        "batch_size": 32,
        "max_tokens": 8000,
        "distributed": torch.cuda.device_count() > 1,
        "num_epochs": 100,
        "accum_iter": 1,
        "base_lr": 1.43,  # peak lr 1e-3 after warmup with rate()
        "max_padding": 256,
        "warmup": 4000,
        "num_layers": 5,
        "d_model": 512,
        "dropout": 0.3,
        "attn_dropout": 0.1,
        "label_smoothing": 0.1,
        "weight_decay": 1e-4,
        "shared_embeddings": True,
        "keep_last": 5,
        "file_prefix": "checkpoint/ne_en/model_",
    }

    if not exists(model_path):
        train_model(vocab_src, vocab_tgt, config, load_fn=load_fn)

    model = make_model(
        len(vocab_src),
        len(vocab_tgt),
        N=config["num_layers"],
        d_model=config["d_model"],
        shared_embeddings=config["shared_embeddings"],
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    return model


if __name__ == "__main__":
    model = load_trained_model()

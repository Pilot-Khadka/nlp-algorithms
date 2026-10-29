import os
import json
import random
import functools
from pathlib import Path

import torch

from nlp_algorithms.encoder_decoder.train import train_model
from nlp_algorithms.encoder_decoder.data import (
    load_vocab,
    load_ne_en,
    load_multi30k,
    load_tatoeba,
)
from nlp_algorithms.encoder_decoder.evaluate import evaluate_bleu, last_checkpoints
from nlp_algorithms.util.general_util import resolve_checkpoint_dir

# dataset config -> load_fn returning (train, valid, test) with src/tgt columns
LOAD_FNS = {
    "ne_en": lambda dataset: load_ne_en,
    "multi30k": lambda dataset: load_multi30k,
    "tatoeba": lambda dataset: functools.partial(
        load_tatoeba, dataset.data_dir, dataset.url
    ),
}


def _subsample(load_fn, max_samples):
    # smaller train set for quick test runs, val and test stay full
    train, val, test = load_fn()
    idx = list(range(len(train)))
    random.Random(42).shuffle(idx)
    return train.select(idx[:max_samples]), val, test


def get_load_fn(config):
    name = config.dataset.name
    if name not in LOAD_FNS:
        raise KeyError(
            f"{name} has no machine translation loader. Available: {list(LOAD_FNS)}"
        )
    load_fn = LOAD_FNS[name](config.dataset)
    max_samples = config.dataset.get("max_samples")
    if max_samples:
        load_fn = functools.partial(_subsample, load_fn, max_samples)
    return load_fn


def build_train_config(config):
    model, train = config.model, config.train
    checkpoint_dir = resolve_checkpoint_dir(config)
    return {
        "batch_size": train.batch_size,
        "max_tokens": train.get("max_tokens"),
        "distributed": train.get("multi_gpu", False) and torch.cuda.device_count() > 1,
        "num_epochs": train.epochs,
        "max_hours": train.get("max_hours"),
        "accum_iter": train.get("accum_iter", 1),
        # multiplier on rate(), the peak lr is base_lr * d_model^-0.5 * warmup^-0.5
        "base_lr": train.learning_rate,
        "max_padding": train.get("max_padding", 128),
        "warmup": train.get("warmup", 4000),
        "num_layers": model.get("num_layers", 6),
        "d_model": model.get("d_model", 512),
        "dropout": model.get("dropout", 0.1),
        "attn_dropout": model.get("attn_dropout"),
        "shared_embeddings": model.get("shared_embeddings", False),
        "label_smoothing": train.get("label_smoothing", 0.1),
        "weight_decay": train.get("weight_decay", 0.01),
        "amp": train.get("amp", True),
        "keep_last": train.get("keep_last"),
        "log_every": train.get("log_every", 1000),
        "file_prefix": os.path.join(checkpoint_dir, "model_"),
    }


def get_vocab_path(config):
    default = os.path.join(resolve_checkpoint_dir(config), "vocab.pt")
    return config.tokenizer.get("vocab_path", default)


def get_vocab(config, load_fn):
    tokenizer = config.tokenizer
    joint = tokenizer.get("joint", False)
    if config.model.get("shared_embeddings", False) and not joint:
        raise ValueError(
            "shared_embeddings needs a joint vocab (tokenizer.joint: true)"
        )

    vocab_kwargs = {"joint": joint}
    if tokenizer.get("sample_size"):
        vocab_kwargs["sample_size"] = tokenizer.sample_size
    return load_vocab(
        load_fn=load_fn,
        vocab_path=get_vocab_path(config),
        vocab_size=tokenizer.get("vocab_size", 8192),
        **vocab_kwargs,
    )


def run_machine_translation(config):
    load_fn = get_load_fn(config)
    train_config = build_train_config(config)
    checkpoint_dir = os.path.dirname(train_config["file_prefix"])
    os.makedirs(checkpoint_dir, exist_ok=True)

    vocab_src, vocab_tgt = get_vocab(config, load_fn)

    print("config:", train_config, flush=True)
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(train_config, f, indent=2)

    train_model(vocab_src, vocab_tgt, train_config, load_fn=load_fn)
    return evaluate_machine_translation(config)


def evaluate_machine_translation(config, checkpoint=None, split="test"):
    load_fn = get_load_fn(config)
    train_config = build_train_config(config)
    checkpoint_dir = os.path.dirname(train_config["file_prefix"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    if not os.path.exists(get_vocab_path(config)):
        raise FileNotFoundError(f"no vocab at {get_vocab_path(config)}, train first")
    vocab_src, vocab_tgt = get_vocab(config, load_fn)

    task = config.task
    beam_size = task.get("beam_size", 5)
    if checkpoint is not None:
        runs = {
            "greedy": ([checkpoint], 1),
            f"beam{beam_size}": ([checkpoint], beam_size),
        }
    else:
        prefix = train_config["file_prefix"]
        runs = {
            "best_greedy": ([f"{prefix}best.pt"], 1),
            f"best_beam{beam_size}": ([f"{prefix}best.pt"], beam_size),
        }
        last = last_checkpoints(prefix, train_config["keep_last"] or 0)
        if train_config["keep_last"] and last:
            print("averaging:", last, flush=True)
            runs[f"avg_last_beam{beam_size}"] = (last, beam_size)

    _, valid_rows, test_rows = load_fn()
    rows = test_rows if split == "test" else valid_rows
    if task.get("eval_rows"):
        rows = rows.select(range(task.eval_rows))

    scores, outputs = evaluate_bleu(
        train_config,
        vocab_src,
        vocab_tgt,
        rows,
        runs,
        len_penalty=task.get("len_penalty", 1.2),
    )

    # bleu.json / translations.txt for the training run, a suffix otherwise
    suffix = f"_{Path(checkpoint).stem}" if checkpoint is not None else ""
    suffix += "" if split == "test" else f"_{split}"
    with open(os.path.join(checkpoint_dir, f"bleu{suffix}.json"), "w") as f:
        json.dump(scores, f, indent=2)
    with open(os.path.join(checkpoint_dir, f"translations{suffix}.txt"), "w") as f:
        for i, row in enumerate(rows):
            f.write(f"src : {row['src']}\nref : {row['tgt']}\n")
            for name in runs:
                f.write(f"{name:15s}: {outputs[name][i]}\n")
            f.write("\n")
    return scores

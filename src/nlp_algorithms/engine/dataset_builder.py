from typing import List


import os
import gc
import inspect
from tqdm import tqdm
from collections import Counter

from nlp_algorithms.engine.registry import (
    get_from_registry,
    DATA_READER_REGISTRY,
    DOWNLOADER_REGISTRY,
    TOKENIZER_REGISTRY,
)
from nlp_algorithms.infra.vocabulary import Vocabulary
from nlp_algorithms.util.general_util import resolve_tokenizer_path


class DatasetBundle:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        token_vocab,
        label_vocab=None,
        src_vocab=None,
        tgt_vocab=None,
        src_tokenizer=None,
        tgt_tokenizer=None,
    ):
        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset
        self.vocab = token_vocab
        self.label_vocab = label_vocab
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer


def _is_trainable_tokenizer(tokenizer) -> bool:
    return hasattr(tokenizer, "train") and callable(getattr(tokenizer, "train"))


def _is_trained(tokenizer) -> bool:
    """check if tokenizer has already been trained."""
    if hasattr(tokenizer, "vocab") and tokenizer.vocab:
        return True
    if hasattr(tokenizer, "merges") and tokenizer.merges:
        return True
    return False


def _collect_corpus(dataset, key: str) -> str:
    texts: List[str] = []

    try:
        total = len(dataset)
    except TypeError:
        total = None

    iterator = tqdm(range(len(dataset)), total=total, desc=f"Collecting '{key}' corpus")

    for i in iterator:
        item = dataset[i]
        val = item[key]

        if isinstance(val, str):
            texts.append(val)
        elif isinstance(val, list):
            texts.extend(val)
        else:
            raise ValueError(f"Unsupported type for key '{key}': {type(val)}")

    return "\n".join(texts)


def build_vocab_from_key(
    dataset,
    config,
    key: str,
    tokenizer,
    vocab_size: int = 10000,
    min_freq: int = 1,
    special_tokens=None,
):
    """
    For trainable tokenizers (BPE): Trains on the corpus and extracts vocab directly
    For stateless tokenizers (Whitespace): Counts tokens and builds vocab from frequencies
    """
    if special_tokens is None:
        special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }

    if tokenizer is not None and _is_trainable_tokenizer(tokenizer):
        checkpoint_dir = getattr(dataset, "checkpoint_dir", "./checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = resolve_tokenizer_path(config, key)

        if os.path.exists(save_path):
            print(f"Loading saved tokenizer from: {save_path}")
            tokenizer.load(save_path)

        elif not _is_trained(tokenizer):
            print(f"Training tokenizer on '{key}' corpus...")
            corpus = _collect_corpus(dataset, key)
            tokenizer.train(corpus=corpus)
            del corpus
            gc.collect()

            print(f"Saving tokenizer to: {save_path}")
            tokenizer.save(path=save_path)

        else:
            print(f"Using pre-trained tokenizer vocab for '{key}'")

        token_to_id = dict(special_tokens)
        next_id = len(special_tokens)

        if hasattr(tokenizer, "vocab"):
            for token in tokenizer.vocab:
                if token not in token_to_id:
                    token_to_id[token] = next_id
                    next_id += 1
                    if len(token_to_id) >= vocab_size:
                        break

        return Vocabulary(token_to_id)

    counter = Counter()

    for item in tqdm(dataset, desc=f"Building vocab from '{key}'"):
        val = item[key]

        if tokenizer is not None:
            if isinstance(val, str):
                counter.update(tokenizer.tokenize(val))
            elif isinstance(val, list):
                for text in val:
                    counter.update(tokenizer.tokenize(text))
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(val)}")
        else:
            if isinstance(val, list):
                counter.update(val)
            else:
                counter.update([val])

    token_to_id = dict(special_tokens)
    idx = len(token_to_id)

    for token, freq in counter.most_common(vocab_size - len(special_tokens)):
        if freq >= min_freq and token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    del counter
    gc.collect()

    return Vocabulary(token_to_id)


class DatasetBundleBuilder:
    def build(self, config):
        if not config.dataset.name:
            raise RuntimeError("config.dataset.name is not defined")

        if not config.tokenizer.name:
            raise RuntimeError("config.tokenizer.name is not defined")

        if not config.dataset.vocab_size:
            raise RuntimeError("config.dataset.voacab_size is not defined")

        data_downloader_cls = get_from_registry(
            DOWNLOADER_REGISTRY, config.dataset.name
        )
        data_dir = data_downloader_cls().download_and_prepare(config)

        data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name)

        max_samples = config.dataset.get("max_samples", None)

        train = data_reader_cls(
            data_dir=data_dir,
            split="train",
            max_samples=max_samples,
        )

        available_splits = set(os.listdir(data_dir))

        has_valid = "valid" in available_splits or "validation" in available_splits
        has_test = "test" in available_splits

        if has_valid:
            valid = data_reader_cls(
                data_dir=data_dir,
                split="valid",
                max_samples=max_samples,
            )
        else:
            valid = data_reader_cls(
                data_dir=data_dir,
                split="test",
                max_samples=max_samples,
            )

        if has_test:
            test = data_reader_cls(
                data_dir=data_dir,
                split="test",
                max_samples=max_samples,
            )
        else:
            # fallback to valid (rare but symmetric)
            test = valid

        tokenizer_cls = get_from_registry(TOKENIZER_REGISTRY, config.tokenizer.name)

        tokenizer_kwargs = {}
        sig = inspect.signature(tokenizer_cls.__init__)

        if "vocab_size" in sig.parameters:
            bpe_vocab_size = getattr(
                config.tokenizer, "vocab_size", config.dataset.vocab_size
            )
            tokenizer_kwargs["vocab_size"] = bpe_vocab_size

        tokenizer = tokenizer_cls(**tokenizer_kwargs)
        tgt_tokenizer = None

        label_vocab = None
        token_vocab = None
        src_vocab = None
        tgt_vocab = None

        # vocabulary build logic
        if config.task.name in {"classification", "ner"}:
            token_vocab = build_vocab_from_key(
                train,
                config=config,
                key="text",
                tokenizer=tokenizer,
                vocab_size=config.dataset.vocab_size,
            )
        elif config.task.name == "translation":
            src_vocab = build_vocab_from_key(
                train,
                config=config,
                key="src",
                tokenizer=tokenizer,
                vocab_size=config.dataset.vocab_size,
            )
            tokenizer_tgt = tokenizer_cls(**tokenizer_kwargs)
            tgt_vocab = build_vocab_from_key(
                train,
                config=config,
                key="tgt",
                tokenizer=tokenizer_tgt,
                vocab_size=config.dataset.vocab_size,
            )
        else:
            token_vocab = build_vocab_from_key(
                train,
                config=config,
                key="text",
                tokenizer=tokenizer,
                vocab_size=config.dataset.vocab_size,
            )

        return DatasetBundle(
            train_dataset=train,
            val_dataset=valid,
            test_dataset=test,
            token_vocab=token_vocab,
            label_vocab=label_vocab,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            src_tokenizer=tokenizer,
            tgt_tokenizer=tgt_tokenizer,
        )

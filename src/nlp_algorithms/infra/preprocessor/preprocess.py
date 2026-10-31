"""
downloader -> reader  -> preprocessor(tokenize+encode) -> dataloader -> collator(batchify/pad) -> model
"""

import os
import re
import sys
import time
import logging
import functools
import multiprocessing

import torch
from tqdm import tqdm

from nlp_algorithms.util.logger import fmt_time

logger = logging.getLogger(__name__)

_TASK_REGISTRY = {}


def register(task_name):
    def decorator(cls):
        _TASK_REGISTRY[task_name] = cls
        return cls

    return decorator


@register("classification")
class ClassificationBuilder:
    def build(self, example):
        return {"input_ids": example["text"], "label": example["label"]}


@register("language_modeling")
class LanguageModelingBuilder:
    def build(self, example):
        return {"input_ids": example["input_ids"]}


@register("masked_language_modeling")
class MaskedLanguageModelingBuilder:
    # nsp needs sentence boundaries, so every sentence is encoded on its own
    def build(self, example):
        text = re.sub(r"<br\s*/?>", " ", example["text"])
        sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        return {"sentences": sentences}


@register("translation")
class TranslationBuilder:
    def build(self, example):
        # builder returns raw strings (src/tgt)
        # tokenization happens in PreprocessedDataset
        return {"input_ids": example["src"], "labels": example["tgt"]}


def _encode(texts, tokenizer, vocab, max_len):
    return [vocab.encode(tokenizer.tokenize(text))[:max_len] for text in texts]


def encode_texts(texts, tokenizer, vocab, max_len, num_workers=1):
    # bpe is pure python, split the texts over processes. the workers return lists,
    # many small tensors through a pool would each need a shared memory file
    # the bar only in a terminal, piped logs (kaggle) get a line every quarter
    show_bar = sys.stderr.isatty()
    if num_workers <= 1 or len(texts) < 1000:
        texts = tqdm(texts, desc="Tokenizing", disable=not show_bar)
        return _encode(texts, tokenizer, vocab, max_len)

    size = -(-len(texts) // (num_workers * 20))
    chunks = [texts[i : i + size] for i in range(0, len(texts), size)]
    fn = functools.partial(_encode, tokenizer=tokenizer, vocab=vocab, max_len=max_len)
    encoded = []
    with multiprocessing.get_context("fork").Pool(num_workers) as pool:
        done = tqdm(
            pool.imap(fn, chunks),
            total=len(chunks),
            desc="Tokenizing",
            disable=not show_bar,
        )
        for i, chunk in enumerate(done, 1):
            encoded.extend(chunk)
            if not show_bar and i % max(1, len(chunks) // 4) == 0 and i < len(chunks):
                logger.info(f"encoded {len(encoded)}/{len(texts)} texts")
    return encoded


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        src_tokenizer,
        vocab,
        task,
        max_len,
        target_vocab=None,
        tgt_tokenizer=None,
        num_workers=1,
    ):
        self.task = task
        self.builder = _TASK_REGISTRY[task]()
        self.max_len = max_len

        is_translation = task == "translation"

        if is_translation:
            if tgt_tokenizer is None:
                raise ValueError("tgt_tokenizer must be provided for translation task.")
            if target_vocab is None:
                raise ValueError("target_vocab must be provided for translation task.")

        specs = [
            self.builder.build(item)
            for item in tqdm(
                base_dataset,
                desc=f"Building {task} dataset",
                disable=not sys.stderr.isatty(),
            )
        ]
        # every source text in one list, so they can be encoded in parallel
        texts = []
        for task_spec in specs:
            if "input_ids" in task_spec:
                texts.append(task_spec["input_ids"])
            texts.extend(task_spec.get("sentences", []))
        start = time.perf_counter()
        logger.info(f"encoding {len(texts)} texts with {num_workers} processes")
        encoded_texts = encode_texts(texts, src_tokenizer, vocab, max_len, num_workers)
        num_tokens = sum(len(ids) for ids in encoded_texts)
        logger.info(
            f"encoded {len(texts)} texts, {num_tokens / 1e6:.1f}M tokens in "
            f"{fmt_time(time.perf_counter() - start)} ({num_workers} processes)"
        )
        encoded_texts = iter(encoded_texts)

        self.examples = []
        for task_spec in specs:
            example = {}

            if "input_ids" in task_spec:
                encoded = next(encoded_texts)
                example["input_ids"] = torch.tensor(encoded, dtype=torch.long)

            if "sentences" in task_spec:
                example["sentences"] = [
                    torch.tensor(next(encoded_texts), dtype=torch.long)
                    for _ in task_spec["sentences"]
                ]

            if "labels" in task_spec:
                if is_translation:
                    assert tgt_tokenizer is not None
                    assert target_vocab is not None
                    tokens = tgt_tokenizer.tokenize(task_spec["labels"])
                    encoded = target_vocab.encode(tokens)[:max_len]
                else:
                    raise ValueError(
                        f"Task '{task}' produced string labels but no tgt_tokenizer is set."
                    )

                example["labels"] = torch.tensor(encoded, dtype=torch.long)

            if "label" in task_spec:
                example["label"] = task_spec["label"]

            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def __iter__(self):
        for i in range(len(self.examples)):
            yield self.examples[i]

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.examples, path)
        logger.info(f"saved {len(self.examples)} examples to {path}")

    @classmethod
    def load(cls, path):
        examples = torch.load(path)
        dataset = cls.__new__(cls)  # bypass __init__
        dataset.examples = examples
        return dataset

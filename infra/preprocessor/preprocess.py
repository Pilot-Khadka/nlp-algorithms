"""
downloader -> reader  -> preprocessor(tokenize+encode) -> dataloader -> collator(batchify/pad) -> model
"""

import torch
from tqdm import tqdm

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


@register("translation")
class TranslationBuilder:
    def build(self, example):
        # builder returns raw strings (src/tgt)
        # tokenization happens in PreprocessedDataset
        return {"input_ids": example["src"], "labels": example["tgt"]}


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

        self.examples = []
        for item in tqdm(base_dataset, desc=f"Building {task} dataset"):
            task_spec = self.builder.build(item)
            example = {}

            if "input_ids" in task_spec:
                tokens = src_tokenizer.tokenize(task_spec["input_ids"])
                encoded = vocab.encode(tokens)[:max_len]
                example["input_ids"] = torch.tensor(encoded, dtype=torch.long)

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

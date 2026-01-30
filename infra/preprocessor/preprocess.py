"""
downloader -> reader  -> preprocessor(tokenize+encode) -> dataloader -> collator(batchify/pad) -> model
"""

import torch

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
        return {"input_ids": example["src"], "labels": example["tgt"]}


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        tokenizer,
        vocab,
        task,
        max_len,
        target_vocab=None,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer()
        self.vocab = vocab
        self.target_vocab = target_vocab
        self.max_len = max_len
        self.task = task

        if task not in _TASK_REGISTRY:
            raise ValueError(
                f"Unknown task: {task}. Available tasks: {list(_TASK_REGISTRY.keys())}"
            )
        self.builder = _TASK_REGISTRY[task]()

        if task == "translation" and target_vocab is None:
            raise ValueError("Translation task requires target_vocab to be provided")

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]

        task_spec = self.builder.build(item)
        result = {}

        if "input_ids" in task_spec:
            text = task_spec["input_ids"]
            tokens = self.tokenizer.tokenize(text)
            encoded = self.vocab.encode(tokens)[: self.max_len]
            result["input_ids"] = encoded

        if "labels" in task_spec:
            assert self.target_vocab is not None
            label_text = task_spec["labels"]
            label_tokens = self.tokenizer.tokenize(label_text)
            label_encoded = self.target_vocab.encode(label_tokens)[: self.max_len]
            result["labels"] = label_encoded

        if "label" in task_spec:
            result["label"] = task_spec["label"]

        return result

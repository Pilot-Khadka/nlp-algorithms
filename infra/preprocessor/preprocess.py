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
        self.task = task
        self.builder = _TASK_REGISTRY[task]()
        self.max_len = max_len

        self.examples = []
        tokenizer = tokenizer()
        for item in tqdm(base_dataset, desc=f"Building {task} dataset"):
            task_spec = self.builder.build(item)
            example = {}
            if "input_ids" in task_spec:
                tokens = tokenizer.tokenize(task_spec["input_ids"])
                encoded = torch.tensor(vocab.encode(tokens)[:max_len], dtype=torch.long)
                example["input_ids"] = encoded
            if "labels" in task_spec:
                assert target_vocab is not None
                tokens = tokenizer.tokenize(task_spec["labels"])
                encoded = torch.tensor(
                    target_vocab.encode(tokens)[:max_len], dtype=torch.long
                )
                example["labels"] = encoded
            if "label" in task_spec:
                example["label"] = task_spec["label"]
            self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

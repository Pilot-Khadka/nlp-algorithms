"""
downloader -> reader  -> preprocessor(tokenize+encode) -> dataloader -> collator(batchify/pad) -> model
"""

import torch


class ClassificationBuilder:
    def build(self, example):
        return {"input_ids": example["input_ids"], "label": example["label"]}


class LanguageModelingBuilder:
    def build(self, example):
        return {"input_ids": example["input_ids"]}


class PreprocessedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        tokenizer,
        vocab,
        task,
        max_len=128,
    ):
        self.base = base_dataset
        self.tokenizer = tokenizer()
        self.vocab = vocab
        self.max_len = max_len
        self.task = task

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        text = item["text"]
        label = item["label"]

        tokens = self.tokenizer.tokenize(text)
        encoded = self.vocab.encode(tokens)[: self.max_len]

        return {
            "input_ids": encoded,
            "label": label,
        }

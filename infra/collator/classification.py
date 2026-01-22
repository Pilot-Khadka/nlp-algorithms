from typing import List, Dict, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from infra.collator import BaseCollator
from engine.registry import register_collator


@register_collator("classification")
class ClassificationCollator(BaseCollator):
    def __init__(
        self,
        tokenizer,
        vocab,
        max_len: int = 128,
        architecture: str = "transformer",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.max_len = max_len
        self.architecture = architecture

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        texts = [item["text"] for item in batch]
        labels = [item["label"] for item in batch]

        tokenized = [self.tokenizer.tokenize(text) for text in texts]
        encoded = [self.vocab.encode(tokens)[: self.max_len] for tokens in tokenized]

        if self.architecture == "rnn":
            return self._collate_rnn(encoded, labels)
        else:
            return self._collate_transformer(encoded, labels)

    def _collate_rnn(
        self, encoded: List[List[int]], labels: List[int]
    ) -> Dict[str, Tensor]:
        lengths = [len(seq) for seq in encoded]

        # descending for pack_padded_sequence
        sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        encoded = [encoded[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        lengths = [lengths[i] for i in sorted_indices]

        sequences = [torch.tensor(seq, dtype=torch.long) for seq in encoded]
        input_ids = pad_sequence(
            sequences, batch_first=True, padding_value=self.vocab.pad_id
        )

        return {
            "input_ids": input_ids,
            "lengths": torch.tensor(lengths, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def _collate_transformer(
        self, encoded: List[List[int]], labels: List[int]
    ) -> Dict[str, Tensor]:
        sequences = [torch.tensor(seq, dtype=torch.long) for seq in encoded]
        input_ids = pad_sequence(
            sequences, batch_first=True, padding_value=self.vocab.pad_id
        )

        attention_mask = (input_ids != self.vocab.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from dataset.downloader import ImdbDownloader
    from dataset.reader import IMDBDataset
    from core_tokenization import WhitespaceTokenizer
    from infra.vocabulary import Vocabulary

    cfg = {
        "dataset": {
            "name": "imdb",
            "data_dir": "../../dataset/dataset_imdb/",
            "url": "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        }
    }

    data_dir = ImdbDownloader.download_and_prepare(cfg)
    train_dataset = IMDBDataset(data_dir=data_dir, split="train")
    test_dataset = IMDBDataset(data_dir=data_dir, split="test")

    example = test_dataset[0]
    print(example)

    tokenizer = WhitespaceTokenizer()
    all_tokens = []
    for item in train_dataset:
        all_tokens.extend(tokenizer.tokenize(item["text"]))

    vocab = Vocabulary.from_tokens(tokens=all_tokens, vocab_size=10000, min_freq=1)

    print("Vocab size:", len(vocab))

    collator = ClassificationCollator(
        tokenizer=tokenizer,
        vocab=vocab,
        max_len=128,
        architecture="transformer",
    )

    loader = DataLoader(
        test_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator.collate,
    )

    batch = next(iter(loader))

    print("\n=== BATCH ===")
    print("input_ids:", batch["input_ids"].shape)
    print(batch["input_ids"])
    print("attention_mask:", batch["attention_mask"].shape)
    print("labels:", batch["labels"])

from typing import List, Dict, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from nlp_algorithms.infra.collator import BaseCollator
from nlp_algorithms.engine.registry import register_collator


def _to_long_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().clone().long()
    return torch.tensor(x, dtype=torch.long)


@register_collator("classification")
class ClassificationCollator(BaseCollator):
    def __init__(self, vocab, architecture="transformer"):
        super().__init__()
        self.vocab = vocab
        self.architecture = architecture

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        encoded = [item["input_ids"] for item in batch]
        labels = [item["label"] for item in batch]

        if self.architecture == "transformer":
            return self._collate_transformer(encoded, labels)
        else:
            return self._collate_rnn(encoded, labels)

    def _collate_rnn(self, encoded, labels):
        lengths = [len(seq) for seq in encoded]

        sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        encoded = [encoded[i] for i in sorted_indices]
        labels = [labels[i] for i in sorted_indices]
        lengths = [lengths[i] for i in sorted_indices]

        sequences = [_to_long_tensor(x) for x in encoded]
        input_ids = pad_sequence(
            sequences, batch_first=True, padding_value=self.vocab.pad_id
        )

        return {
            "input_ids": input_ids,
            "lengths": _to_long_tensor(lengths),
            "labels": _to_long_tensor(labels),
        }

    def _collate_transformer(self, encoded, labels):
        sequences = [_to_long_tensor(x) for x in encoded]
        input_ids = pad_sequence(
            sequences, batch_first=True, padding_value=self.vocab.pad_id
        )

        attention_mask = (input_ids != self.vocab.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": _to_long_tensor(labels),
        }


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from nlp_algorithms.dataset.downloader import ImdbDownloader
    from nlp_algorithms.dataset.reader import IMDBDataset
    from nlp_algorithms.tokenization import WhitespaceTokenizer
    from nlp_algorithms.infra.vocabulary import Vocabulary

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

    # pyrefly: ignore [missing-attribute]
    vocab = Vocabulary.from_tokens(tokens=all_tokens, vocab_size=10000, min_freq=1)

    print("Vocab size:", len(vocab))

    collator = ClassificationCollator(
        vocab=vocab,
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

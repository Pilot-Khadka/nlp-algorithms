from typing import List, Dict, Any

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from infra.collator import BaseCollator

from engine.registry import register_collator


@register_collator("language_modeling")
class LanguageModelingCollator(BaseCollator):
    """Collator for language modeling (next token prediction)."""

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

        tokenized = [self.tokenizer.tokenize(text) for text in texts]
        encoded = [self.vocab.encode(tokens)[: self.max_len] for tokens in tokenized]

        if self.architecture == "rnn":
            return self._collate_rnn(encoded)
        else:
            return self._collate_transformer(encoded)

    def _collate_rnn(self, encoded: List[List[int]]) -> Dict[str, Tensor]:
        lengths = [len(seq) for seq in encoded]

        sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        encoded = [encoded[i] for i in sorted_indices]
        lengths = [lengths[i] for i in sorted_indices]

        # Input: all tokens except last
        # Labels: all tokens except first
        input_seqs = [
            torch.tensor(seq[:-1], dtype=torch.long) for seq in encoded if len(seq) > 1
        ]
        label_seqs = [
            torch.tensor(seq[1:], dtype=torch.long) for seq in encoded if len(seq) > 1
        ]

        input_ids = pad_sequence(
            input_seqs, batch_first=True, padding_value=self.vocab.pad_id
        )
        labels = pad_sequence(label_seqs, batch_first=True, padding_value=-100)

        adjusted_lengths = [max(1, l - 1) for l in lengths]

        return {
            "input_ids": input_ids,
            "lengths": torch.tensor(adjusted_lengths, dtype=torch.long),
            "labels": labels,
        }

    def _collate_transformer(self, encoded: List[List[int]]) -> Dict[str, Tensor]:
        # Input: all tokens except last
        # Labels: all tokens except first (teacher forcing)
        input_seqs = [
            torch.tensor(seq[:-1], dtype=torch.long) for seq in encoded if len(seq) > 1
        ]
        label_seqs = [
            torch.tensor(seq[1:], dtype=torch.long) for seq in encoded if len(seq) > 1
        ]

        input_ids = pad_sequence(
            input_seqs, batch_first=True, padding_value=self.vocab.pad_id
        )
        labels = pad_sequence(label_seqs, batch_first=True, padding_value=-100)

        attention_mask = (input_ids != self.vocab.pad_id).long()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from dataset.reader import PTBDataset
    from dataset.downloader import PTBDownloader
    from core_tokenization import WhitespaceTokenizer
    from infra.vocabulary import Vocabulary

    cfg = {
        "dataset": {
            "data_dir": "../../dataset/dataset_penn_treebank/",
            "train_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
            "valid_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
            "test_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
        }
    }

    data_dir = PTBDownloader.download_and_prepare(cfg)
    train_dataset = PTBDataset(data_dir, split="train")
    test_dataset = PTBDataset(data_dir, split="test")

    tokenizer = WhitespaceTokenizer()
    all_tokens = []
    for item in train_dataset:
        all_tokens.extend(tokenizer.tokenize(item["text"]))

    vocab = Vocabulary.from_tokens(tokens=all_tokens, vocab_size=10000, min_freq=1)

    print("Vocab size:", len(vocab))
    print("ID of 'the':", vocab.token_to_id.get("the"))
    print("First 10 vocab items:", list(vocab.token_to_id.items())[:10])

    encoded = vocab.encode(all_tokens[:20])
    decoded = vocab.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)

    collator = LanguageModelingCollator(tokenizer, vocab, max_len=20)
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

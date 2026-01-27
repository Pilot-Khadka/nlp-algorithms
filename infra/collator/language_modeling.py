from typing import List, Dict, Any, Optional

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from infra.collator import BaseCollator
# from engine.registry import register_collator


# @register_collator("language_modeling")
class LanguageModelingCollator(BaseCollator):
    def __init__(
        self,
        vocab,
        architecture: str = "transformer",
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
    ):
        super().__init__()
        self.vocab = vocab
        self.architecture = architecture

        if batch_size is None or seq_len is None:
            raise ValueError(
                "corpus_mode=True requires batch_size and seq_len parameters"
            )
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.corpus_data = None  # will hold batchified data
        self.position = 0

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        encoded = [item["input_ids"] for item in batch]
        if self.architecture == "transformer":
            return self._collate_transformer(encoded)
        else:
            return self._collate_rnn(encoded)

    def _collate_rnn(self, encoded: List[List[int]]) -> Dict[str, Tensor]:
        """description: RNN collation for independent sequences."""
        lengths = [len(seq) for seq in encoded]

        sorted_indices = sorted(
            range(len(lengths)), key=lambda i: lengths[i], reverse=True
        )
        encoded = [encoded[i] for i in sorted_indices]
        lengths = [lengths[i] for i in sorted_indices]

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
        """description: Transformer collation for independent sequences."""
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

    collator = LanguageModelingCollator(vocab=vocab, seq_len=20)
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

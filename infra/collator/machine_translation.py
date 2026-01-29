from typing import List, Any, Dict

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from infra.collator import BaseCollator

from engine.registry import register_collator


@register_collator("translation")
class TranslationCollator(BaseCollator):
    def __init__(
        self,
        src_tokenizer,
        src_vocab,
        tgt_tokenizer,
        tgt_vocab,
        max_len: int = 128,
        architecture: str = "transformer",
    ):
        super().__init__()
        self.src_tokenizer = src_tokenizer
        self.src_vocab = src_vocab
        self.tgt_tokenizer = tgt_tokenizer
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.architecture = architecture

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        src_texts = [item["src"] for item in batch]
        tgt_texts = [item["tgt"] for item in batch]

        src_tokenized = [self.src_tokenizer.tokenize(text) for text in src_texts]
        tgt_tokenized = [self.tgt_tokenizer.tokenize(text) for text in tgt_texts]

        src_encoded = [
            self.src_vocab.encode(tokens)[: self.max_len] for tokens in src_tokenized
        ]
        tgt_encoded = [
            self.tgt_vocab.encode(tokens)[: self.max_len] for tokens in tgt_tokenized
        ]

        if self.architecture == "rnn":
            return self._collate_rnn(src_encoded, tgt_encoded)
        else:
            return self._collate_transformer(src_encoded, tgt_encoded)

    def _collate_rnn(
        self, src_encoded: List[List[int]], tgt_encoded: List[List[int]]
    ) -> Dict[str, Tensor]:
        src_lengths = [len(seq) for seq in src_encoded]
        tgt_lengths = [len(seq) for seq in tgt_encoded]

        # Sort by source length for encoder efficiency
        sorted_indices = sorted(
            range(len(src_lengths)), key=lambda i: src_lengths[i], reverse=True
        )
        src_encoded = [src_encoded[i] for i in sorted_indices]
        tgt_encoded = [tgt_encoded[i] for i in sorted_indices]
        src_lengths = [src_lengths[i] for i in sorted_indices]
        tgt_lengths = [tgt_lengths[i] for i in sorted_indices]

        src_seqs = [torch.tensor(seq, dtype=torch.long) for seq in src_encoded]
        tgt_input_seqs = [
            torch.tensor(seq[:-1], dtype=torch.long)
            for seq in tgt_encoded
            if len(seq) > 1
        ]
        tgt_label_seqs = [
            torch.tensor(seq[1:], dtype=torch.long)
            for seq in tgt_encoded
            if len(seq) > 1
        ]

        src_ids = pad_sequence(
            src_seqs, batch_first=True, padding_value=self.src_vocab.pad_id
        )
        tgt_ids = pad_sequence(
            tgt_input_seqs, batch_first=True, padding_value=self.tgt_vocab.pad_id
        )
        tgt_labels = pad_sequence(tgt_label_seqs, batch_first=True, padding_value=-100)

        adjusted_tgt_lengths = [max(1, l - 1) for l in tgt_lengths]

        return {
            "src_ids": src_ids,
            "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
            "tgt_ids": tgt_ids,
            "tgt_lengths": torch.tensor(adjusted_tgt_lengths, dtype=torch.long),
            "labels": tgt_labels,
        }

    def _collate_transformer(
        self, src_encoded: List[List[int]], tgt_encoded: List[List[int]]
    ) -> Dict[str, Tensor]:
        src_seqs = [torch.tensor(seq, dtype=torch.long) for seq in src_encoded]
        tgt_input_seqs = [
            torch.tensor(seq[:-1], dtype=torch.long)
            for seq in tgt_encoded
            if len(seq) > 1
        ]
        tgt_label_seqs = [
            torch.tensor(seq[1:], dtype=torch.long)
            for seq in tgt_encoded
            if len(seq) > 1
        ]

        src_ids = pad_sequence(
            src_seqs, batch_first=True, padding_value=self.src_vocab.pad_id
        )
        tgt_ids = pad_sequence(
            tgt_input_seqs, batch_first=True, padding_value=self.tgt_vocab.pad_id
        )
        tgt_labels = pad_sequence(tgt_label_seqs, batch_first=True, padding_value=-100)

        src_mask = (src_ids != self.src_vocab.pad_id).long()
        tgt_mask = (tgt_ids != self.tgt_vocab.pad_id).long()

        return {
            "src_ids": src_ids,
            "src_mask": src_mask,
            "tgt_ids": tgt_ids,
            "tgt_mask": tgt_mask,
            "labels": tgt_labels,
        }


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    from dataset.downloader import TatoebaDownloader
    from dataset.reader import TatoebaDataset
    from core_tokenization import WhitespaceTokenizer
    from infra.vocabulary import Vocabulary

    cfg = {
        "dataset": {
            "url": "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eng-nep.tar",
            "data_dir": "../../dataset/dataset_tatoeba_eng_nep/",
            "vocab_size": 10000,
        },
    }

    data_dir = TatoebaDownloader.download_and_prepare(cfg)
    train_dataset = TatoebaDataset(data_dir, split="train")
    test_dataset = TatoebaDataset(data_dir, split="test")

    src_tokenizer = WhitespaceTokenizer()
    tgt_tokenizer = WhitespaceTokenizer()

    src_tokens, tgt_tokens = [], []
    for item in train_dataset:
        src_tokens.extend(src_tokenizer.tokenize(item["src"]))
        tgt_tokens.extend(tgt_tokenizer.tokenize(item["tgt"]))

    src_vocab = Vocabulary.from_tokens(tokens=src_tokens, vocab_size=10000, min_freq=1)
    tgt_vocab = Vocabulary.from_tokens(tokens=tgt_tokens, vocab_size=10000, min_freq=1)

    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(tgt_vocab))

    collator = TranslationCollator(
        src_tokenizer=src_tokenizer,
        src_vocab=src_vocab,
        tgt_tokenizer=tgt_tokenizer,
        tgt_vocab=tgt_vocab,
        max_len=20,
    )

    loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collator.collate
    )

    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("Source IDs shape:", batch["src_ids"].shape)
    print("Target IDs shape:", batch["tgt_ids"].shape)
    print("Labels shape:", batch["labels"].shape)

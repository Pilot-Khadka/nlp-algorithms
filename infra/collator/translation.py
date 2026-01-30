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
        pad_id: int = 0,
        label_pad_id: int = 0,
        architecture: str = "transformer",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.label_pad_id = label_pad_id
        self.architecture = architecture

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        if "pad_id" in batch[0]:
            self.pad_id = batch[0]["pad_id"]

        src_encoded = [item["input_ids"] for item in batch]
        tgt_encoded = [item["labels"] for item in batch]

        if self.architecture == "transformer":
            return self._collate_transformer(src_encoded, tgt_encoded)
        else:
            return self._collate_rnn(src_encoded, tgt_encoded)

    def _collate_rnn(
        self, src_encoded: List[List[int]], tgt_encoded: List[List[int]]
    ) -> Dict[str, Tensor]:
        src_lengths = [len(s) for s in src_encoded]
        tgt_lengths = [len(t) for t in tgt_encoded]

        sorted_indices = sorted(
            range(len(src_lengths)), key=lambda i: src_lengths[i], reverse=True
        )
        src_encoded = [src_encoded[i] for i in sorted_indices]
        tgt_encoded = [tgt_encoded[i] for i in sorted_indices]
        src_lengths = [src_lengths[i] for i in sorted_indices]
        tgt_lengths = [tgt_lengths[i] for i in sorted_indices]

        src_seqs = [torch.tensor(seq, dtype=torch.long) for seq in src_encoded]
        tgt_input = [torch.tensor(seq[:-1], dtype=torch.long) for seq in tgt_encoded]
        tgt_label = [torch.tensor(seq[1:], dtype=torch.long) for seq in tgt_encoded]

        src_ids = pad_sequence(src_seqs, batch_first=True, padding_value=self.pad_id)
        tgt_ids = pad_sequence(tgt_input, batch_first=True, padding_value=self.pad_id)
        tgt_labels = pad_sequence(
            tgt_label, batch_first=True, padding_value=self.label_pad_id
        )

        tgt_adj_lengths = [max(1, l - 1) for l in tgt_lengths]

        return {
            "src_ids": src_ids,
            "src_lengths": torch.tensor(src_lengths, dtype=torch.long),
            "tgt_ids": tgt_ids,
            "tgt_lengths": torch.tensor(tgt_adj_lengths, dtype=torch.long),
            "labels": tgt_labels,
        }

    def _collate_transformer(
        self, src_encoded: List[List[int]], tgt_encoded: List[List[int]]
    ) -> Dict[str, Tensor]:
        src_seqs = [torch.tensor(seq, dtype=torch.long) for seq in src_encoded]
        tgt_input = [torch.tensor(seq[:-1], dtype=torch.long) for seq in tgt_encoded]
        tgt_label = [torch.tensor(seq[1:], dtype=torch.long) for seq in tgt_encoded]

        src_ids = pad_sequence(src_seqs, batch_first=True, padding_value=self.pad_id)
        tgt_ids = pad_sequence(tgt_input, batch_first=True, padding_value=self.pad_id)
        tgt_labels = pad_sequence(
            tgt_label, batch_first=True, padding_value=self.label_pad_id
        )

        src_mask = (src_ids != self.pad_id).long()
        tgt_mask = (tgt_ids != self.pad_id).long()

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

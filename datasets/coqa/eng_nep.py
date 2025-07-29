"""
english to nepali tar file downloaded from:
https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm


def read_parallel_data(src_path, trg_path, limit=None):
    with (
        open(src_path, "r", encoding="utf-8") as f_src,
        open(trg_path, "r", encoding="utf-8") as f_trg,
    ):
        src_lines = f_src.readlines()
        trg_lines = f_trg.readlines()

    if limit:
        src_lines = src_lines[:limit]
        trg_lines = trg_lines[:limit]

    paired = [
        (s.strip(), t.strip())
        for s, t in zip(src_lines, trg_lines)
        if s.strip() and t.strip()
    ]
    return paired


class TranslationDataset(Dataset):
    def __init__(self, data, pad_idx):
        self.data = data
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        src_batch, trg_batch = zip(*batch)
        src_lens = [len(s) for s in src_batch]
        trg_lens = [len(t) for t in trg_batch]

        max_src_len = max(src_lens)
        max_trg_len = max(trg_lens)

        src_padded = [s + [self.pad_idx] * (max_src_len - len(s)) for s in src_batch]
        trg_padded = [t + [self.pad_idx] * (max_trg_len - len(t)) for t in trg_batch]

        return torch.tensor(src_padded), torch.tensor(trg_padded)


if __name__ == "__main__":
    spm.SentencePieceTrainer.train(
        input=["dataset/train.src", "dataset/train.trg"],
        model_prefix="spm",
        vocab_size=10000,
        character_coverage=1.0,
    )
    sp = spm.SentencePieceProcessor(model_file="spm.model")

    train_data = read_parallel_data(
        "dataset/train.src", "dataset/train.trg", limit=50000
    )

    pad_id = sp.pad_id()
    dataset = TranslationDataset(train_data, sp, pad_id)
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn
    )

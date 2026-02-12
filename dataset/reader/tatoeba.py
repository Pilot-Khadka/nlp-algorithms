from typing import Optional

import os
import torch
import numpy as np
from tqdm import tqdm

from engine.registry import register_reader


def build_or_load_index(text_file, max_samples=None, desc="Indexing"):
    idx_file = text_file + ".idx"

    if os.path.exists(idx_file):
        data = np.load(idx_file)
        if max_samples is not None:
            data = data[:max_samples]
        return data

    offsets = []
    with open(text_file, "r", encoding="utf-8") as f:
        pbar = tqdm(desc=desc)
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(offset)
            pbar.update(1)

            if max_samples is not None and len(offsets) >= max_samples:
                break
        pbar.close()

    offsets_np = np.array(offsets, dtype=np.int64)
    np.save(idx_file, offsets_np)
    return offsets_np


@register_reader("tatoeba")
class TatoebaDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        max_samples: Optional[int] = None,
    ):
        self.split = split
        self.data_dir = data_dir
        self.src_file = os.path.join(data_dir, f"{split}.src")
        self.trg_file = os.path.join(data_dir, f"{split}.trg")

        self.src_offsets = build_or_load_index(
            self.src_file, max_samples, "Indexing src"
        )
        self.trg_offsets = build_or_load_index(
            self.trg_file, max_samples, "Indexing trg"
        )

        self.n = len(self.src_offsets)

        self._src_f = open(self.src_file, "r", encoding="utf-8", newline="")
        self._trg_f = open(self.trg_file, "r", encoding="utf-8", newline="")

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        self._src_f.seek(int(self.src_offsets[index]))
        self._trg_f.seek(int(self.trg_offsets[index]))
        return {
            "src": self._src_f.readline().rstrip("\n"),
            "tgt": self._trg_f.readline().rstrip("\n"),
        }

    def __del__(self):
        try:
            self._src_f.close()
            self._trg_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    from dataset.downloader.tatoeba import TatoebaDownloader

    cfg = {
        "dataset": {
            "url": "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eng-nep.tar",
            "data_dir": "../dataset_tatoeba_eng_nep/",
            "vocab_size": 10000,
        },
    }

    data_dir = TatoebaDownloader.download_and_prepare(cfg)
    test_dataset = TatoebaDataset(data_dir, split="train")

    example = test_dataset[0]
    print(example)

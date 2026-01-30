import os

import torch

from engine.registry import register_reader


@register_reader("tatoeba")
class TatoebaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, split: str):
        self.split = split
        self.data_dir = data_dir
        self.src_file = os.path.join(data_dir, f"{split}.src")
        self.trg_file = os.path.join(data_dir, f"{split}.trg")

        self.src_offsets = []
        self.trg_offsets = []

        with open(self.src_file, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                self.src_offsets.append(offset)
                offset += len(line.encode("utf-8"))

        with open(self.trg_file, "r", encoding="utf-8") as f:
            offset = 0
            for line in f:
                self.trg_offsets.append(offset)
                offset += len(line.encode("utf-8"))

    def __len__(self):
        return len(self.src_offsets)

    def __getitem__(self, index):
        with (
            open(self.src_file, "r", encoding="utf-8") as src_f,
            open(self.trg_file, "r", encoding="utf-8") as trg_f,
        ):
            src_f.seek(self.src_offsets[index])
            trg_f.seek(self.trg_offsets[index])
            src_line = src_f.readline().strip()
            trg_line = trg_f.readline().strip()
            return {"src": src_line, "tgt": trg_line}


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

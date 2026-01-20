from typing import Dict, List

import os
from tqdm import tqdm

from torch.utils.data import Dataset


class TatoebaDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
    ):
        self.split = split
        self.data_dir = data_dir

        prepared_flag = os.path.join(data_dir, ".prepared")
        if not os.path.exists(prepared_flag):
            raise RuntimeError(
                f"Data not prepared in {data_dir}. "
                f"Call TatoebaDownloader.download_and_prepare(cfg) first."
            )

        self.examples = self._load_raw_data()

    def _load_raw_data(self) -> List[Dict[str, str]]:
        src_file = os.path.join(self.data_dir, f"{self.split}.src")
        trg_file = os.path.join(self.data_dir, f"{self.split}.trg")

        if not os.path.exists(src_file) or not os.path.exists(trg_file):
            raise FileNotFoundError(
                f"Split '{self.split}' not found in {self.data_dir}. "
                f"Expected {src_file} and {trg_file}"
            )

        examples = []
        with open(src_file, "r", encoding="utf-8") as src_f:
            with open(trg_file, "r", encoding="utf-8") as trg_f:
                total = sum(1 for _ in open(src_file, "r", encoding="utf-8"))
                for src_line, trg_line in tqdm(
                    zip(src_f, trg_f), desc=f"Loading {self.split} data", total=total
                ):
                    examples.append({"src": src_line.strip(), "tgt": trg_line.strip()})

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.examples[index]


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
    test_dataset = TatoebaDataset(data_dir, split="test")

    example = test_dataset[0]
    print(example)

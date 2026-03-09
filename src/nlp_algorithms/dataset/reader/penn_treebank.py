from typing import Dict

import os
from torch.utils.data import Dataset

from nlp_algorithms.engine.registry import register_reader


@register_reader("ptb")
class PTBDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split

        if not os.path.exists(os.path.join(data_dir, ".prepared")):
            raise RuntimeError(
                f"Data not prepared in {data_dir}. "
                f"Call PTBDownloader.download_and_prepare(cfg) first."
            )

        self.text = self._load_text()

    def _load_text(self) -> str:
        all_files = os.listdir(self.data_dir)

        target_file = None
        for f_name in all_files:
            if self.split.lower() in f_name.lower() and f_name.endswith(".txt"):
                target_file = f_name
                break

        if target_file is None:
            raise FileNotFoundError(
                f"Could not find a .txt file containing '{self.split}' in {self.data_dir}. "
                f"Files found: {all_files}"
            )

        file_path = os.path.join(self.data_dir, target_file)

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Split '{self.split}' not found. Expected {file_path}"
            )

        with open(file_path, "r", encoding="utf-8") as f:
            # replace newlines with <eos> and split into tokens
            text = f.read().replace("\n", " <eos> ")

        print(f"Loaded {len(text)} characters from {self.split} split")
        return text

    def __len__(self) -> int:
        return 1  # one text document for this split

    def __getitem__(self, index: int) -> Dict[str, str]:
        if index != 0:
            raise IndexError("PTBDataset contains a single text sample")
        return {"text": self.text}


if __name__ == "__main__":
    from ..downloader import PTBDownloader

    cfg = {
        "dataset": {
            "data_dir": "../dataset_penn_treebank/",
            "train_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
            "valid_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
            "test_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
        }
    }

    data_dir = PTBDownloader.download_and_prepare(cfg)

    # train_dataset = PTBDataset(data_dir, split="train")
    # valid_dataset = PTBDataset(data_dir, split="valid")
    test_dataset = PTBDataset(data_dir, split="test")

    sample = test_dataset[0]
    raw_text = sample["text"]

    print("raw text:", raw_text)
    print(f"Dataset size: {len(test_dataset)} text")

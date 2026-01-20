from typing import List

import os

from torch.utils.data import Dataset


class PTBDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split

        if not os.path.exists(os.path.join(data_dir, ".prepared")):
            raise RuntimeError(
                f"Data not prepared in {data_dir}. "
                f"Call PTBDownloader.download_and_prepare(cfg) first."
            )

        self.tokens = self._load_raw_tokens()

    def _load_raw_tokens(self) -> List[str]:
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
            tokens = text.split()

        print(f"Loaded {len(tokens)} tokens from {self.split} split")
        return tokens

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> str:
        return self.tokens[index]

    def get_all_tokens(self) -> List[str]:
        return self.tokens


if __name__ == "__main__":
    from dataset.downloader import PTBDownloader

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

    token = test_dataset[0]
    all_tokens = test_dataset.get_all_tokens()

    print(f"Dataset size: {len(test_dataset)} tokens")
    print(f"First 10 tokens: {all_tokens[:10]}")

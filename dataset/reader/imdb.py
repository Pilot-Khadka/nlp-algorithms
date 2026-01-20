import os
from typing import Dict, List
from torch.utils.data import Dataset
from tqdm import tqdm


class IMDBDataset(Dataset):
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split

        prepared_flag = os.path.join(data_dir, ".prepared")
        if not os.path.exists(prepared_flag):
            raise RuntimeError(
                f"IMDB data not prepared in {data_dir}. "
                f"Call IMDBDownloader.download_and_prepare(cfg) first."
            )

        self.examples = self._load_raw_data()

    def _load_raw_data(self) -> List[Dict[str, str]]:
        split_dir = os.path.join(self.data_dir, self.split)
        pos_path = os.path.join(split_dir, "pos")
        neg_path = os.path.join(split_dir, "neg")

        if not os.path.exists(pos_path) or not os.path.exists(neg_path):
            raise FileNotFoundError(
                f"Split '{self.split}' not found in {self.data_dir}. "
                f"Expected {pos_path} and {neg_path}"
            )

        examples = []
        files = []
        for label, folder in [(1, pos_path), (0, neg_path)]:
            for file in sorted(os.listdir(folder)):
                if file.endswith(".txt"):
                    files.append((label, os.path.join(folder, file)))

        for label, path in tqdm(files, desc=f"Loading IMDB {self.split}"):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip().replace("\n", " ")
                examples.append(
                    {
                        "text": text,
                        "label": label,
                        "path": path,
                    }
                )

        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.examples[index]


if __name__ == "__main__":
    from dataset.downloader import ImdbDownloader

    cfg = {
        "dataset": {
            "data_dir": "../dataset_imdb/",
            "url": "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            "archive_name": "aclImdb_v1.tar.gz",
        }
    }

    data_dir = ImdbDownloader.download_and_prepare(cfg)
    test_dataset = IMDBDataset(data_dir, split="test")

    example = test_dataset[0]
    print(example)

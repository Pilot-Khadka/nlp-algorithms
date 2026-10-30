from typing import Dict, List, Optional

import os
import glob
import polars as pl
from torch.utils.data import Dataset


from nlp_algorithms.engine.registry import register_reader


@register_reader("imdb")
class IMDBDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        max_samples: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.split = split

        prepared_flag = os.path.join(data_dir, ".prepared")
        if not os.path.exists(prepared_flag):
            raise RuntimeError(
                f"IMDB data not prepared in {data_dir}. "
                f"Call ImdbDownloader.download_and_prepare(cfg) first."
            )

        examples = self._load_raw_data()
        if max_samples is not None:
            examples = examples[:max_samples]

        self.examples = examples

    def _load_raw_data(self) -> List[Dict[str, str]]:
        # "train+unsupervised" reads both
        files = []
        for split in self.split.split("+"):
            found = sorted(glob.glob(os.path.join(self.data_dir, f"{split}-*.parquet")))
            if not found:
                raise FileNotFoundError(
                    f"Split '{split}' not found in {self.data_dir}. "
                    f"Expected {split}-*.parquet"
                )
            files.extend(found)

        # text, label (0 neg, 1 pos, -1 for the unsupervised reviews)
        df = pl.concat([pl.read_parquet(path) for path in files])
        return [
            {"text": text.strip().replace("\n", " "), "label": label}
            for text, label in zip(df["text"], df["label"])
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.examples[index]


if __name__ == "__main__":
    from ..downloader import ImdbDownloader

    cfg = {
        "dataset": {
            "data_dir": "../dataset_imdb/",
            "repo": "stanfordnlp/imdb",
        }
    }

    data_dir = ImdbDownloader.download_and_prepare(cfg)
    test_dataset = IMDBDataset(data_dir, split="test")

    example = test_dataset[0]
    print(example)

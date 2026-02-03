"""
github link:https://github.com/sharad461/nepali-translator?tab=readme-ov-file https://github.com/sharad461/nepali-translator?tab=readme-ov-file
benchmark: https://lt4all.elra.info/proceedings/lt4all2019/pdf/2019.lt4all-1.94.pdf
"""

from typing import Optional

import os
import polars as pl
from torch.utils.data import Dataset

from engine.registry import register_reader


@register_reader("eng_nep")
class EngNepDataset(Dataset):
    def __init__(self, data_dir: str, split: str, max_samples: Optional[int] = None):
        self.split = split
        self.data_dir = data_dir
        self.max_samples = max_samples

        if not os.path.exists(os.path.join(data_dir, ".prepared")):
            raise RuntimeError(
                f"Data not prepared in {data_dir}. "
                f"Call PTBDownloader.download_and_prepare(cfg) first."
            )

        self.df = self._load_text()

        if self.max_samples is not None:
            if self.max_samples < len(self.df):
                self.df = self.df.head(self.max_samples)

    def _load_flores_split(self, split_dir: str):
        en_file = None
        np_file = None
        for f in os.listdir(split_dir):
            if f.endswith(".jsonl"):
                if "eng_Latn" in f:
                    en_file = f
                elif "npi_Deva" in f:
                    np_file = f

        if en_file is None or np_file is None:
            raise RuntimeError(
                f"Missing Flores files in {split_dir}. Need eng_Latn + npi_Deva."
            )

        df_en = pl.read_ndjson(os.path.join(split_dir, en_file)).select(["id", "text"])
        df_np = pl.read_ndjson(os.path.join(split_dir, np_file)).select(["id", "text"])

        df = df_en.join(df_np.rename({"text": "tgt"}), on="id", how="inner").rename(
            {"text": "src"}
        )

        return df

    def _load_text(self):
        if self.split == "train":
            train_dir = os.path.join(self.data_dir, "data")
            parquet_files = [f for f in os.listdir(train_dir) if f.endswith(".parquet")]

            if len(parquet_files) == 0:
                raise RuntimeError("No parquet files found in train directory.")

            parquet_path = os.path.join(train_dir, parquet_files[0])
            df = pl.read_parquet(parquet_path)

            # expecting "en", "ne" columns in the 208k dataset
            if "en" not in df.columns or "ne" not in df.columns:
                raise RuntimeError("Train parquet must contain 'en' and 'ne' columns.")

            df = df.rename({"en": "src", "ne": "tgt"})
            return df

        elif self.split in ["dev", "valid"]:
            dev_dir = os.path.join(self.data_dir, "dev")
            return self._load_flores_split(dev_dir)

        elif self.split == "test":
            test_dir = os.path.join(self.data_dir, "devtest")
            return self._load_flores_split(test_dir)

        else:
            raise ValueError(f"Unknown split: {self.split}")

    def __len__(self):
        return self.df.height

    def __getitem__(self, index):
        row = self.df.row(index, named=True)
        return {"src": row["src"], "tgt": row["tgt"]}

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


if __name__ == "__main__":
    from dataset.downloader import HuggingFaceDatasetDownloader

    downloader = HuggingFaceDatasetDownloader()
    config = {
        "dataset": {
            "name": "huggingface",
            "data_dir": "../dataset_hf_eng_nep",
            "repos": [
                {
                    "id": "sharad461/ne-en-parallel-208k",
                    "files": "*",  # or omit "files" to download all
                },
                {
                    "id": "openlanguagedata/flores_plus",
                    "files": ["dev/npi_Deva.jsonl", "devtest/npi_Deva.jsonl"],
                },
            ],
        }
    }

    data_dir = downloader.download_and_prepare(config)

    train_dataset = EngNepDataset(data_dir=data_dir, split="train")
    valid_dataset = EngNepDataset(data_dir=data_dir, split="valid")
    test_dataset = EngNepDataset(data_dir=data_dir, split="test")

    sample = train_dataset[0]

    print("raw text:", sample["src"])
    print("target text:", sample["tgt"])
    print(f"Dataset size: {len(train_dataset)} text")

    sample = test_dataset[0]

    print("raw text:", sample["src"])
    print("target text:", sample["tgt"])
    print(f"Dataset size: {len(test_dataset)} text")

    sample = valid_dataset[0]

    print("raw text:", sample["src"])
    print("target text:", sample["tgt"])
    print(f"Dataset size: {len(valid_dataset)} text")

from typing import Dict, Any, Optional, Tuple, List

import os
import shutil
import tarfile
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset

from dataset.base import DatasetBundle, DatasetUtils

# later will be tokenizer agnostic
from core_tokenization import WhitespaceTokenizer


class TatoebaDataset(Dataset):
    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str,
        source_vocab: Optional[Dict[str, int]] = None,
        target_vocab: Optional[Dict[str, int]] = None,
        max_len: int = 128,
    ):
        self.dataset_url = cfg["dataset"]["url"]
        self.archive_name = os.path.basename(self.dataset_url)
        self.extract_name = self.archive_name.rsplit(".tar", 1)[0]
        self.data_dir = cfg["dataset"]["data_dir"]
        self.archive_path = os.path.join(self.data_dir, self.archive_name)
        self.split = split
        self.max_len = max_len
        self.tokenizer = WhitespaceTokenizer()

        self.source_vocab = source_vocab
        self.target_vocab = target_vocab

        self.prepare_data()

        self.source_texts, self.target_texts = self.load_raw_data()

        # build vocab if not already done
        if not self.tokenizer.get_vocab():
            all_texts = self.source_texts + self.target_texts
            vocab_size = cfg["dataset"].get("vocab_size", 10000)
            self.tokenizer.build_vocab(all_texts, vocab_size)

        self.source_ids = [
            self.tokenizer.encode(text, max_len=self.max_len)
            for text in tqdm(self.source_texts, desc="Encoding source")
        ]
        self.target_ids = [
            self.tokenizer.encode(text, max_len=self.max_len)
            for text in tqdm(self.target_texts, desc="Encoding target")
        ]

    def prepare_data(self):
        os.makedirs(self.data_dir, exist_ok=True)

        files_needed = {f"{self.split}.src", f"{self.split}.trg"}
        existing_files = set(os.listdir(self.data_dir))

        if not files_needed.issubset(existing_files):
            self._download_and_extract()

    def _download_and_extract(self):
        temp_archive = self.archive_path + ".tmp"
        temp_extract_dir = os.path.join(self.data_dir, "tmp_extract")

        try:
            if not os.path.exists(self.archive_path):
                print("Downloading Tatoeba dataset...")
                DatasetUtils.download_file(self.dataset_url, temp_archive)
                os.rename(temp_archive, self.archive_path)

            print("Extracting...")
            os.makedirs(temp_extract_dir, exist_ok=True)
            with tarfile.open(self.archive_path, "r:*") as tar:
                tar.extractall(path=temp_extract_dir)

            self._flatten_extracted_files(temp_extract_dir, self.data_dir)
            shutil.rmtree(temp_extract_dir, ignore_errors=True)

        except Exception as e:
            print("Extraction failed:", e)
            if os.path.exists(temp_archive):
                os.remove(temp_archive)
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            raise

    def _flatten_extracted_files(self, src_root: str, dst_root: Optional[str] = None):
        import gzip

        if dst_root is None:
            dst_root = src_root

        DatasetUtils.ensure_dir(dst_root)

        for root, _, files in os.walk(src_root):
            for file in tqdm(files, desc=f"Processing files in {root}"):
                src_path = os.path.join(root, file)

                if os.path.abspath(root) == os.path.abspath(dst_root):
                    continue

                dst_name = file
                dst_path = os.path.join(dst_root, dst_name)

                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(dst_name)
                    count = 1
                    while os.path.exists(dst_path):
                        dst_name = f"{base}_{count}{ext}"
                        dst_path = os.path.join(dst_root, dst_name)
                        count += 1

                if file.endswith(".gz"):
                    decompressed_path = os.path.join(dst_root, dst_name[:-3])

                    try:
                        with (
                            gzip.open(src_path, "rt", encoding="utf-8") as f_in,
                            open(decompressed_path, "w", encoding="utf-8") as f_out,
                        ):
                            shutil.copyfileobj(f_in, f_out)

                        os.remove(src_path)

                    except Exception:
                        raise
                else:
                    shutil.move(src_path, dst_path)

        for root, dirs, files in os.walk(src_root, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                if os.path.isdir(dir_path):
                    try:
                        os.rmdir(dir_path)
                    except OSError:
                        pass

    def load_raw_data(self) -> Tuple[List[str], List[str]]:
        self.prepare_data()
        dataset_path = os.path.join(self.data_dir)
        src_file = os.path.join(dataset_path, f"{self.split}.src")
        trg_file = os.path.join(dataset_path, f"{self.split}.trg")

        source_texts, target_texts = [], []
        with (
            open(src_file, "r", encoding="utf-8") as src_f,
            open(trg_file, "r", encoding="utf-8") as trg_f,
        ):
            for src_line, trg_line in tqdm(
                zip(src_f, trg_f),
                desc=f"Loading {self.split} data",
                total=sum(1 for _ in open(src_file, "r", encoding="utf-8")),
            ):
                source_texts.append(src_line.strip())
                target_texts.append(trg_line.strip())
        return source_texts, target_texts

    def __len__(self):
        return len(self.source_ids)

    def __getitem__(self, index):
        return {
            "source": torch.tensor(self.source_ids[index], dtype=torch.long),
            "target": torch.tensor(self.target_ids[index], dtype=torch.long),
        }


def get_tatoeba_dataloaders(cfg):
    batch_size = cfg["train"]["batch_size"]

    train_dataset = TatoebaDataset(cfg, split="train")
    dev_dataset = TatoebaDataset(
        cfg,
        split="dev",
        source_vocab=train_dataset.source_vocab,
        target_vocab=train_dataset.target_vocab,
    )
    test_dataset = TatoebaDataset(
        cfg,
        split="test",
        source_vocab=train_dataset.source_vocab,
        target_vocab=train_dataset.target_vocab,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate
    )
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=pad_collate)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, collate_fn=pad_collate
    )

    return DatasetBundle(
        train_loader,
        dev_loader,
        test_loader,
        vocab={
            "source": train_dataset.source_vocab,
            "target": train_dataset.target_vocab,
        },
    )


def pad_collate(batch):
    # Pad sequences to the max length in the batch
    src_batch = [item["source"] for item in batch]
    trg_batch = [item["target"] for item in batch]

    src_batch = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=0
    )
    trg_batch = torch.nn.utils.rnn.pad_sequence(
        trg_batch, batch_first=True, padding_value=0
    )

    return {"source": src_batch, "target": trg_batch}


if __name__ == "__main__":
    cfg = {
        "dataset": {
            "url": "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eng-nep.tar",
            "data_dir": "dataset_tatoeba_eng_nep/",
        },
        "train": {"batch_size": 32},
    }
    loaders = get_tatoeba_dataloaders(cfg)
    print(loaders)

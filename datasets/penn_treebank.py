from typing import Dict, Any, Optional, Tuple, List

import os
import shutil
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.base import DatasetBundle
from datasets.base import DatasetUtils
from utils.utils import get_num_workers


class IMDBDataset(Dataset):
    def __init__(self, cfg, split: str, vocab: Optional[Dict[str, int]] = None):
        self.data_dir = cfg.datasets["data_dir"]
        self.dataset_url = cfg.datasets["url"]
        self.archive_name = cfg.datasets["archive_name"]
        self.extract_name = self.archive_name.split(".")[0].replace("_v1", "")
        self.split = split

        self.min_freq = cfg.datasets.get("min_freq", 1)
        self.max_seq_len = cfg.datasets.get("max_seq_len", 256)

        self.prepare_data()
        texts, labels = self.load_raw_data()

        if vocab is None and split == "train":
            self.vocab = self.build_vocab(texts)
        elif vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError(
                "Vocabulary must be provided for non-train splits "
                "or build from train split first"
            )

        self.inputs, self.labels, self.doc_ids, self.chunk_ids = self._encode_and_chunk(
            texts, labels
        )

    def _encode_and_chunk(
        self, texts: List[str], labels: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Tokenize + numericalize once
        - Chunk long docs into fixed-size segments
        - Pre-pad so we avoid per-batch pad_sequence
        Returns:
            inputs:  (N, max_seq_len) int64
            labels:  (N,) int64
            doc_ids: (N,) int64
            chunk_ids: (N,) int64
        """
        vocab_get = self.vocab.get
        unk_id = self.vocab["<unk>"]
        pad_id = self.vocab["<pad>"]
        max_len = self.max_seq_len

        all_inputs = []
        all_labels = []
        all_doc_ids = []
        all_chunk_ids = []

        for doc_id, (text, label) in enumerate(zip(texts, labels)):
            tokens = self.tokenize(text)
            token_ids = [vocab_get(tok, unk_id) for tok in tokens]

            if max_len is None:
                chunks = [token_ids]
            else:
                chunks = [
                    token_ids[i : i + max_len]
                    for i in range(0, len(token_ids), max_len)
                ]

            for chunk_id, chunk in enumerate(chunks):
                if max_len is not None:
                    if len(chunk) < max_len:
                        chunk = chunk + [pad_id] * (max_len - len(chunk))
                    elif len(chunk) > max_len:
                        chunk = chunk[:max_len]

                all_inputs.append(chunk)
                all_labels.append(label)
                all_doc_ids.append(doc_id)
                all_chunk_ids.append(chunk_id)

        inputs_tensor = torch.tensor(all_inputs, dtype=torch.long)  # (N, L)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long)  # (N,)
        doc_ids_tensor = torch.tensor(all_doc_ids, dtype=torch.long)  # (N,)
        chunk_ids_tensor = torch.tensor(all_chunk_ids, dtype=torch.long)  # (N,)

        return inputs_tensor, labels_tensor, doc_ids_tensor, chunk_ids_tensor

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
            "doc_ids": self.doc_ids[index],
            "chunk_ids": self.chunk_ids[index],
        }

    def get_special_tokens(self) -> Dict[str, int]:
        return {"<pad>": 0, "<unk>": 1}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self.tokenize(text))

        vocab = self.get_special_tokens()
        idx = len(vocab)

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in vocab:
                vocab[word] = idx
                idx += 1

        return vocab

    def prepare_data(self):
        extract_path = os.path.join(self.data_dir, self.extract_name)

        if os.path.exists(extract_path) and self._is_complete_extraction(extract_path):
            return

        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self._download_and_extract()

    def _download_and_extract(self):
        import tarfile

        archive_filename = self.archive_name
        extract_name = self.extract_name
        final_archive = os.path.join(self.data_dir, archive_filename)
        extract_path = os.path.join(self.data_dir, extract_name)
        temp_archive = final_archive + ".tmp"

        try:
            if not os.path.exists(final_archive):
                print("Downloading IMDb dataset...")
                DatasetUtils.download_file(self.dataset_url, temp_archive)
                DatasetUtils.ensure_dir(self.data_dir)
                os.rename(temp_archive, final_archive)

            print("Extracting...")
            with tarfile.open(final_archive, "r:gz") as tar:
                tar.extractall(path=self.data_dir, filter="data")

            extracted_dir = os.path.join(self.data_dir, "aclImdb")
            if extracted_dir != extract_path and os.path.exists(extracted_dir):
                os.rename(extracted_dir, extract_path)

            if os.path.exists(final_archive):
                os.remove(final_archive)

        except (KeyboardInterrupt, Exception):
            if os.path.exists(temp_archive):
                os.remove(temp_archive)
            if os.path.exists(extract_path) and not self._is_complete_extraction(
                extract_path
            ):
                shutil.rmtree(extract_path, ignore_errors=True)
            raise

    def _is_complete_extraction(self, path: str) -> bool:
        for split in ["train", "test"]:
            split_dir = os.path.join(path, split)
            if not os.path.exists(split_dir):
                return False
            for label_dir in ["pos", "neg"]:
                if not os.path.exists(os.path.join(split_dir, label_dir)):
                    return False
        return True

    def load_raw_data(self) -> Tuple[List[str], List[int]]:
        split_dir = os.path.join(self.data_dir, self.extract_name, self.split)
        pos_path = os.path.join(split_dir, "pos")
        neg_path = os.path.join(split_dir, "neg")

        texts, labels = [], []
        for label, folder in [(1, pos_path), (0, neg_path)]:
            for file in sorted(os.listdir(folder)):  # sorted for reproducibility
                if file.endswith(".txt"):
                    with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                        text = f.read().strip().replace("\n", " ")
                        texts.append(text)
                        labels.append(label)

        return texts, labels

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def num_classes(self) -> int:
        return 2


def collate_fixed(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    doc_ids = torch.stack([b["doc_ids"] for b in batch], dim=0)
    chunk_ids = torch.stack([b["chunk_ids"] for b in batch], dim=0)

    lengths = torch.full((input_ids.size(0),), input_ids.size(1), dtype=torch.long)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "lengths": lengths,
        "doc_ids": doc_ids,
        "chunk_ids": chunk_ids,
    }


def get_imdb_dataloaders(cfg):
    batch_size = cfg.train.batch_size

    train_dataset = IMDBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()
    test_dataset = IMDBDataset(cfg, split="test", vocab=vocab)
    num_workers = get_num_workers(cfg)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fixed,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fixed,
        num_workers=num_workers,
        pin_memory=True,
    )

    return DatasetBundle(train_loader, test_loader, test_loader, vocab=vocab)

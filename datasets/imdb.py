from typing import Dict, Any, Optional, Tuple, List

import os
import shutil
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.base import DatasetBundle
from datasets.base import DatasetUtils


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

        self._on_gpu = False

    def move_to_device(self, device: torch.device):
        if not self._on_gpu or self.inputs.device != device:
            self.inputs = self.inputs.to(device, non_blocking=True)
            self.labels = self.labels.to(device, non_blocking=True)
            self.doc_ids = self.doc_ids.to(device, non_blocking=True)
            self.chunk_ids = self.chunk_ids.to(device, non_blocking=True)
            self._on_gpu = True

            self.inputs = self.inputs.contiguous()
            self.labels = self.labels.contiguous()
            self.doc_ids = self.doc_ids.contiguous()
            self.chunk_ids = self.chunk_ids.contiguous()

    def shuffle_inplace(self):
        perm = torch.randperm(len(self), device=self.inputs.device)
        self.inputs = self.inputs[perm]
        self.labels = self.labels[perm]
        self.doc_ids = self.doc_ids[perm]
        self.chunk_ids = self.chunk_ids[perm]

    def _encode_and_chunk(
        self, texts: List[str], labels: List[int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        - Tokenize + numericalize once
        - Chunk long docs into fixed-size segments
        - Pre-allocate tensors for speed
        - Pre-pad so we avoid per-batch pad_sequence
        """
        vocab_get = self.vocab.get
        unk_id = self.vocab["<unk>"]
        pad_id = self.vocab["<pad>"]
        max_len = self.max_seq_len

        total_chunks = 0
        chunk_counts = []
        tokenized_texts = []

        for text in texts:
            tokens = self.tokenize(text)
            tokenized_texts.append(tokens)
            num_chunks = max(1, (len(tokens) + max_len - 1) // max_len)
            chunk_counts.append(num_chunks)
            total_chunks += num_chunks

        inputs_tensor = torch.full((total_chunks, max_len), pad_id, dtype=torch.long)
        labels_tensor = torch.zeros(total_chunks, dtype=torch.long)
        doc_ids_tensor = torch.zeros(total_chunks, dtype=torch.long)
        chunk_ids_tensor = torch.zeros(total_chunks, dtype=torch.long)

        write_idx = 0
        for doc_id, (tokens, label, num_chunks) in enumerate(
            zip(tokenized_texts, labels, chunk_counts)
        ):
            token_ids = [vocab_get(tok, unk_id) for tok in tokens]

            for chunk_id in range(num_chunks):
                start = chunk_id * max_len
                end = min(start + max_len, len(token_ids))
                chunk = token_ids[start:end]

                chunk_len = len(chunk)
                inputs_tensor[write_idx, :chunk_len] = torch.tensor(
                    chunk, dtype=torch.long
                )

                labels_tensor[write_idx] = label
                doc_ids_tensor[write_idx] = doc_id
                chunk_ids_tensor[write_idx] = chunk_id
                write_idx += 1

        return inputs_tensor, labels_tensor, doc_ids_tensor, chunk_ids_tensor

    def __len__(self) -> int:
        return self.inputs.size(0)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.inputs[index],
            self.labels[index],
            self.doc_ids[index],
            self.chunk_ids[index],
        )

    def get_special_tokens(self) -> Dict[str, int]:
        return {"<pad>": 0, "<unk>": 1}

    def tokenize(self, text: str) -> List[str]:
        """tokenization with explicit split"""
        text_lower = text.lower()
        return text_lower.split(" ")

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


def collate_tuple(batch: List[Tuple[torch.Tensor, ...]]) -> Tuple[torch.Tensor, ...]:
    input_ids = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    doc_ids = torch.stack([b[2] for b in batch], dim=0)
    chunk_ids = torch.stack([b[3] for b in batch], dim=0)

    return input_ids, labels, doc_ids, chunk_ids


def get_imdb_dataloaders(cfg):
    batch_size = cfg.train.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = IMDBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()
    test_dataset = IMDBDataset(cfg, split="test", vocab=vocab)

    print(f"Moving datasets to {device}...")
    train_dataset.move_to_device(device)
    test_dataset.move_to_device(device)

    print("Shuffling training data...")
    train_dataset.shuffle_inplace()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tuple,
        num_workers=0,
        pin_memory=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_tuple,
        num_workers=0,
        pin_memory=False,
    )

    return DatasetBundle(train_loader, test_loader, test_loader, vocab=vocab)

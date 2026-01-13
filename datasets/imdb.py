from typing import Dict, Any, Optional, Tuple, List

import os
import shutil
from collections import Counter

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

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
        self.max_seq_len = cfg.datasets.get("max_seq_len", None)

        self.prepare_data()
        self.texts, self.labels = self.load_raw_data()

        if vocab is None and split == "train":
            self.vocab = self.build_vocab(self.texts)
        elif vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError(
                "Vocabulary must be provided for non-train splits or build from train split first"
            )

        self.samples = self._prepare_samples()

    def _prepare_samples(self) -> List[Dict[str, Any]]:
        """
        :token_ids: List[int]
        :label: int
        :doc_id: int (original document index)
        :chunk_id: int (chunk index within document)
        :is_first_chunk: bool
        :is_last_chunk: bool
        """
        samples = []

        for doc_id, (text, label) in enumerate(zip(self.texts, self.labels)):
            tokens = self.tokenize(text)
            token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]

            if self.max_seq_len is None or len(token_ids) <= self.max_seq_len:
                samples.append(
                    {
                        "token_ids": token_ids,
                        "label": label,
                        "doc_id": doc_id,
                        "chunk_id": 0,
                        "is_first_chunk": True,
                        "is_last_chunk": True,
                        "num_chunks": 1,
                    }
                )
            else:
                chunks = [
                    token_ids[i : i + self.max_seq_len]
                    for i in range(0, len(token_ids), self.max_seq_len)
                ]
                num_chunks = len(chunks)

                for chunk_id, chunk in enumerate(chunks):
                    samples.append(
                        {
                            "token_ids": chunk,
                            "label": label,
                            "doc_id": doc_id,
                            "chunk_id": chunk_id,
                            "is_first_chunk": chunk_id == 0,
                            "is_last_chunk": chunk_id == num_chunks - 1,
                            "num_chunks": num_chunks,
                        }
                    )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.samples[index]

    def get_special_tokens(self) -> Dict[str, int]:
        return {"<pad>": 0, "<unk>": 1}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        counter: Counter[str] = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            counter.update(tokens)

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

        temp_archive = archive_filename + ".tmp"

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


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    collaet function preserving chunk information.
    """
    token_ids = [
        torch.tensor(sample["token_ids"], dtype=torch.long) for sample in batch
    ]
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    doc_ids = torch.tensor([sample["doc_id"] for sample in batch], dtype=torch.long)
    chunk_ids = torch.tensor([sample["chunk_id"] for sample in batch], dtype=torch.long)
    is_first_chunk = torch.tensor(
        [sample["is_first_chunk"] for sample in batch], dtype=torch.bool
    )
    is_last_chunk = torch.tensor(
        [sample["is_last_chunk"] for sample in batch], dtype=torch.bool
    )

    padded_tokens = pad_sequence(token_ids, batch_first=True, padding_value=0)
    lengths = torch.tensor([len(t) for t in token_ids], dtype=torch.long)

    return {
        "input_ids": padded_tokens,
        "labels": labels,
        "lengths": lengths,
        "doc_ids": doc_ids,
        "chunk_ids": chunk_ids,
        "is_first_chunk": is_first_chunk,
        "is_last_chunk": is_last_chunk,
    }


def get_imdb_dataloaders(cfg):
    batch_size = cfg.train.batch_size

    train_dataset = IMDBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()
    test_dataset = IMDBDataset(cfg, split="test", vocab=vocab)

    # For chunked data, we should NOT shuffle to keep chunks in order
    shuffle_train = cfg.datasets.get("max_seq_len") is None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_batch,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )

    return DatasetBundle(train_loader, test_loader, test_loader, vocab=vocab)

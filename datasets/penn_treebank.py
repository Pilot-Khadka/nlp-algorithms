from typing import Dict, Optional, List


import os
import requests
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader

from datasets.base import DatasetBundle
from models.model_registry import load_vocab


class PTBDataset(Dataset):
    def __init__(self, cfg, split: str, vocab: Optional[Dict[str, int]] = None):
        self.data_dir = cfg.datasets["data_dir"]
        self.seq_len = cfg.datasets["sequence_length"]
        self.split = split
        self.use_pretrained_embedding = cfg.models.get(
            "use_pretrained_embedding", False
        )

        if not os.path.exists(self.data_dir):
            self.urls = {
                split_name: cfg.datasets[f"{split_name}_url"]
                for split_name in ["train", "valid", "test"]
            }
            self._download_ptb()

        tokens = self._load_tokens()

        if vocab is None:
            if split != "train":
                raise ValueError("Vocabulary must be built from the train split")

            if self.use_pretrained_embedding:
                pretrained_vocab = load_vocab(cfg.model.get("vocab_path"))
                self.vocab = self._add_special_tokens(pretrained_vocab)
            else:
                self.vocab = self._build_vocab(tokens)
        else:
            self.vocab = vocab

        self.encoded = self._encode_tokens(tokens)

    def _load_tokens(self) -> List[str]:
        file_path = os.path.join(self.data_dir, f"{self.split}.txt")
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().replace("\n", " <eos> ").split()

    def _encode_tokens(self, tokens: List[str]) -> torch.Tensor:
        unk_idx = self.vocab["<unk>"]
        vocab_get = self.vocab.get

        encoded = np.fromiter(
            (vocab_get(t, unk_idx) for t in tokens),
            dtype=np.int64,
            count=len(tokens),
        )

        return torch.from_numpy(encoded)

    def _build_vocab(self, tokens: List[str], min_freq: int = 1) -> Dict[str, int]:
        counter = Counter(tokens)
        vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for token, freq in counter.items():
            if freq >= min_freq and token not in vocab:
                vocab[token] = idx
                idx += 1
        return vocab

    def _add_special_tokens(self, pretrained_vocab: Dict) -> Dict[str, int]:
        special_tokens = ["<pad>", "<unk>", "<eos>"]

        word2idx = pretrained_vocab["word2idx"].copy()
        idx2word = pretrained_vocab["idx2word"].copy()
        word_freq = pretrained_vocab.get("word_freq", [])

        max_idx = max(word2idx.values()) if word2idx else -1
        next_idx = max_idx + 1

        for token in special_tokens:
            if token not in word2idx:
                word2idx[token] = next_idx
                idx2word[next_idx] = token

                if isinstance(word_freq, dict):
                    word_freq[token] = 1
                elif isinstance(word_freq, list):
                    while len(word_freq) <= next_idx:
                        word_freq.append(1)
                    word_freq[next_idx] = 1

                print(f"Added special token '{token}' with index {next_idx}")
                next_idx += 1

        return word2idx

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    def _download_ptb(self):
        os.makedirs(self.data_dir, exist_ok=True)

        for split_name, url in self.urls.items():
            file_path = os.path.join(self.data_dir, f"{split_name}.txt")
            if not os.path.exists(file_path):
                print(f"Downloading {split_name}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()

                total_size = int(response.headers.get("content-length", 0))

                with open(file_path, "w", encoding="utf-8") as f:
                    with tqdm(
                        total=total_size,
                        unit="B",
                        unit_scale=True,
                        desc=f"{split_name}.txt",
                    ) as pbar:
                        for chunk in response.iter_content(
                            chunk_size=8192, decode_unicode=True
                        ):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk.encode("utf-8")))

    def __len__(self) -> int:
        return len(self.encoded) - self.seq_len

    # pyrefly: ignore [bad-param-name-override]
    def __getitem__(self, idx: int):
        x = self.encoded[idx : idx + self.seq_len]
        y = self.encoded[idx + 1 : idx + self.seq_len + 1]
        return x, y


def get_num_workers(cfg) -> int:
    import multiprocessing

    num_workers = cfg.datasets.get("num_workers", None)

    if isinstance(num_workers, int) and num_workers > 0:
        return num_workers

    num_cpus = multiprocessing.cpu_count()
    return min(8, max(0, num_cpus - 2))


def get_ptb_dataloaders(cfg):
    batch_size = cfg.train.batch_size
    num_workers = get_num_workers(cfg)
    pin_memory = cfg.datasets.get("pin_memory", torch.cuda.is_available())
    prefetch_factor = cfg.datasets.get("prefetch_factor", 2)

    print(f"DataLoader config: num_workers={num_workers}, pin_memory={pin_memory}")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    train_dataset = PTBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()

    valid_dataset = PTBDataset(cfg, split="valid", vocab=vocab)
    test_dataset = PTBDataset(cfg, split="test", vocab=vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # for the love of god, dont accidentally shuflle it again
        drop_last=True,
        **loader_kwargs,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return DatasetBundle(train_loader, valid_loader, test_loader, vocab=vocab)


if __name__ == "__main__":
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        datasets={
            "data_dir": "../datasets/dataset_penn_treebank/",
            "sequence_length": 10,
            "use_cache": False,
            "num_workers": 0,
            "pin_memory": False,
        },
        models={"use_pretrained_embedding": False},
        train=SimpleNamespace(batch_size=4),
    )

    train_dataset = PTBDataset(cfg, split="train")

    idx = 100
    x, y = train_dataset[idx]

    print("Vocabulary size:", len(train_dataset.get_vocab()))
    print("Input sequence (x):", x)
    print("Target sequence (y):", y)

    idx2word = {idx: word for word, idx in train_dataset.get_vocab().items()}
    # pyrefly: ignore [no-matching-overload]
    x_words = [idx2word.get(i.item(), "<unk>") for i in x]
    # pyrefly: ignore [no-matching-overload]
    y_words = [idx2word.get(i.item(), "<unk>") for i in y]

    print("Input sequence (words):", x_words)
    print("Target sequence (words):", y_words)

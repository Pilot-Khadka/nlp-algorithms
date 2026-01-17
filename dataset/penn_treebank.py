from typing import Dict, Optional, List, Tuple, Iterator

import os
import requests
import numpy as np
from tqdm import tqdm
from collections import Counter

import torch

from dataset.base import DatasetBundle
from model.model_registry import load_vocab


class PTBCorpus:
    def __init__(self, cfg, split: str, vocab: Optional[Dict[str, int]] = None):
        self.data_dir = cfg.dataset["data_dir"]
        self.seq_len = cfg.dataset["sequence_length"]
        self.split = split
        self.use_pretrained_embedding = cfg.model.get("use_pretrained_embedding", False)

        if not os.path.exists(self.data_dir):
            self.urls = {
                split_name: cfg.dataset[f"{split_name}_url"]
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

        self.data = self._encode_tokens(tokens)

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

    def batchify(
        self, batch_size: int, device: Optional[torch.device] = None
    ) -> torch.Tensor:
        """
        Reshape flat data into batch_size parallel streams.

        Example with batch_size=3 and data=[0,1,2,3,4,5,6,7,8,9,10,11]:
            :stream 0: [0, 1, 2, 3]
            :stream 1: [4, 5, 6, 7]
            :stream 2: [8, 9, 10, 11]

            :result shape: (4, 3)
            [[0, 4, 8],
             [1, 5, 9],
             [2, 6, 10],
             [3, 7, 11]]
        """
        n_tokens = self.data.size(0)
        n_batches = n_tokens // batch_size

        data = self.data.narrow(0, 0, n_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()

        if device is not None:
            data = data.to(device)

        return data


class PTBIterator:
    def __init__(
        self,
        data: torch.Tensor,
        seq_len: int,
        device: Optional[torch.device] = None,
        batch_first: bool = True,
    ):
        """
        :data: Batchified tensor of shape (num_steps, batch_size)
        :seq_len: Number of time steps per batch
        :device: Device to place tensors on
        :batch_first: If True, return (batch_size, seq_len) instead of (seq_len, batch_size)
        """
        self.data = data
        self.seq_len = seq_len
        self.device = device
        self.batch_first = batch_first
        self.pos = 0

        self.n_steps = data.size(0)
        self.batch_size = data.size(1)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        self.pos = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pos >= self.n_steps - 1:
            raise StopIteration

        actual_seq_len = min(self.seq_len, self.n_steps - 1 - self.pos)

        x = self.data[self.pos : self.pos + actual_seq_len]
        y = self.data[self.pos + 1 : self.pos + 1 + actual_seq_len]

        self.pos += actual_seq_len

        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)

        if self.batch_first:
            x = x.t().contiguous()  # (batch_size, seq_len)
            y = y.t().contiguous()

        return x, y

    def __len__(self) -> int:
        return (self.n_steps - 1 + self.seq_len - 1) // self.seq_len

    def reset(self):
        self.pos = 0


def get_ptb_dataloaders(cfg):
    batch_size = cfg.train.batch_size
    seq_len = cfg.dataset["sequence_length"]
    batch_first = cfg.dataset.get("batch_first", True)

    if hasattr(cfg.train, "device"):
        device = torch.device(cfg.train.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_corpus = PTBCorpus(cfg, split="train")
    vocab = train_corpus.get_vocab()

    valid_corpus = PTBCorpus(cfg, split="valid", vocab=vocab)
    test_corpus = PTBCorpus(cfg, split="test", vocab=vocab)

    train_data = train_corpus.batchify(batch_size, device)
    valid_data = valid_corpus.batchify(batch_size, device)
    test_data = test_corpus.batchify(batch_size, device)

    print(f"Train data shape: {train_data.shape}")  # (n_steps, batch_size)
    print(f"Valid data shape: {valid_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    print(f"Vocab size: {len(vocab)}")

    train_iter = PTBIterator(train_data, seq_len, device, batch_first)
    valid_iter = PTBIterator(valid_data, seq_len, device, batch_first)
    test_iter = PTBIterator(test_data, seq_len, device, batch_first)

    return DatasetBundle(train_iter, valid_iter, test_iter, vocab=vocab)

import os
import torch
import requests
from tqdm import tqdm
from typing import Dict
from collections import Counter
from torch.utils.data import Dataset, DataLoader

from datasets.base import DatasetBundle
from models.model_registry import load_vocab


class PTBDataset(Dataset):
    def __init__(self, cfg, split, vocab=None):
        self.data_dir = cfg.dataset["data_dir"]
        self.seq_len = cfg.dataset["seq_len"]
        self.split = split
        self.use_pretrained_embedding = cfg.model.get("use_pretrained_embedding", False)

        if not os.path.exists(self.data_dir):
            self.urls = {
                split_name: cfg["dataset"][f"{split_name}_url"]
                for split_name in ["train", "valid", "test"]
            }
            print("File not found, downloading the dataset")
            self.download_ptb()

        self.encoded = self.load_data(split, vocab)
        if vocab is None and split == "train":
            if self.use_pretrained_embedding:
                pretrained_vocab = load_vocab(cfg.model.get("vocab_path"))
                self.vocab = self._ensure_special_tokens(pretrained_vocab)
            else:
                self.vocab = self.build_vocab(self.sentences)
            print(f"Built vocabulary with {len(self.vocab)} tokens")
        elif vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError(
                "Vocabulary must be provided for non-train splits or build from train split first"
            )

    def _ensure_special_tokens(self, pretrained_vocab: Dict) -> Dict[str, int]:
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

    def download_ptb(self):
        os.makedirs(self.data_dir, exist_ok=True)
        for split_name, url in self.urls.items():
            file_path = os.path.join(self.data_dir, f"{split_name}.txt")
            if not os.path.exists(file_path):
                print(f"Downloading {split_name}...")
                response = requests.get(url, stream=True)
                response.raise_for_status()

                # Get total file size for progress bar
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

    def load_data(self, split, vocab=None):
        file_path = os.path.join(self.data_dir, f"{split}.txt")
        with open(file_path, "r") as f:
            text = f.read().replace("\n", "<eos>").split()

        if vocab is None:
            self.vocab = self.build_vocab(text)
            assert max(self.vocab.values()) == len(self.vocab) - 1
        else:
            self.vocab = vocab

        encoded = [self.vocab.get(token, self.vocab["<unk>"]) for token in text]
        return encoded

    def build_vocab(self, tokens, min_freq=1):
        counter = Counter(tokens)
        vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for token, freq in counter.items():
            # also prevent <pad> and <unk> from over-writing
            if freq >= min_freq and token not in vocab:
                vocab[token] = idx
                idx += 1
        return vocab

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx : idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(
            self.encoded[idx + 1 : idx + self.seq_len + 1], dtype=torch.long
        )
        return x, y  # next-token prediction


def get_ptb_dataloaders(cfg):
    batch_size = cfg.dataset.batch_size

    train_dataset = PTBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()

    valid_dataset = PTBDataset(cfg, split="valid", vocab=vocab)
    test_dataset = PTBDataset(cfg, split="test", vocab=vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return DatasetBundle(train_loader, valid_loader, test_loader, vocab=vocab)

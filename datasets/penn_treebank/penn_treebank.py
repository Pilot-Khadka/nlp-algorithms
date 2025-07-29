import os
import torch
import requests
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from datasets.base import DatasetBundle
from tqdm import tqdm


class PTBDataset(Dataset):
    def __init__(self, cfg, split, vocab=None):
        self.data_dir = cfg["data_dir"]
        self.seq_len = cfg["seq_len"]
        self.split = split

        if not os.path.exists(self.data_dir):
            self.urls = {
                split_name: cfg[f"{split_name}_url"]
                for split_name in ["train", "valid", "test"]
            }
            print("File not found, downloading the dataset")
            self.download_ptb()

        self.encoded = self.load_data(split, vocab)

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

        encoded = [self.vocab.get(token, self.vocab["<unk>"])
                   for token in text]
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
        x = torch.tensor(
            self.encoded[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(
            self.encoded[idx + 1: idx + self.seq_len + 1], dtype=torch.long
        )
        return x, y  # next-token prediction


def get_ptb_dataloaders(cfg):
    batch_size = cfg.batch_size

    train_dataset = PTBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()

    valid_dataset = PTBDataset(cfg, split="valid", vocab=vocab)
    test_dataset = PTBDataset(cfg, split="test", vocab=vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return DatasetBundle(train_loader, valid_loader, test_loader, vocab=vocab)

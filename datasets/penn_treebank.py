import os
import torch
import requests
from collections import Counter
from torch.utils.data import Dataset, DataLoader

DATA_DIR = "./ptb_data"
URLS = {
    "train": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
    "valid": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
    "test": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
}

os.makedirs(DATA_DIR, exist_ok=True)


def download_ptb():
    for split, url in URLS.items():
        file_path = os.path.join(DATA_DIR, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"Downloading {split}...")
            r = requests.get(url)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(r.text)


class PTBDataset(Dataset):
    def __init__(self, file_path, vocab=None, seq_len=30):
        with open(file_path, "r") as f:
            text = f.read().replace("\n", "<eos>").split()

        if vocab is None:
            self.vocab = self.build_vocab(text)
            assert max(self.vocab.values()) == len(self.vocab) - 1
        else:
            self.vocab = vocab
        self.encoded = [self.vocab.get(
            token, self.vocab["<unk>"]) for token in text]
        self.seq_len = seq_len

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

    def __len__(self):
        return len(self.encoded) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(
            self.encoded[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(
            self.encoded[idx + 1: idx + self.seq_len + 1], dtype=torch.long
        )
        return x, y  # next-token prediction


def get_ptb_dataloaders(data_dir=None, seq_len=30, batch_size=32):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "ptb_data")

    train_dataset = PTBDataset(os.path.join(
        data_dir, "train.txt"), seq_len=seq_len)
    vocab = train_dataset.vocab  # share vocab
    valid_dataset = PTBDataset(
        os.path.join(data_dir, "valid.txt"), vocab=vocab, seq_len=seq_len
    )
    test_dataset = PTBDataset(
        os.path.join(data_dir, "test.txt"), vocab=vocab, seq_len=seq_len
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, valid_loader, test_loader, vocab

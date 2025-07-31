import os
import tarfile
import requests
from collections import Counter
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from datasets.base import DatasetBundle


class IMDBDataset(Dataset):
    def __init__(self, cfg, split, vocab=None):
        self.data_dir = cfg["data_dir"]
        self.split = split
        self.seq_len = cfg["seq_len"]
        self.dataset_url = cfg["url"]
        self.archive_name = cfg["archive_name"]
        self.extract_name = self.archive_name.split(".")[0].replace("_v1", "")

        if not os.path.exists(self.data_dir):
            self.download_and_extract()

        self.texts, self.labels = self.load_raw_data()
        if vocab is None:
            self.vocab = self.build_vocab(self.texts)
        else:
            self.vocab = vocab

        self.encoded = self.encode_data(self.texts)

    def download_and_extract(self):
        os.makedirs(self.data_dir, exist_ok=True)
        archive_path = os.path.join(self.data_dir, self.archive_name)
        if not os.path.exists(archive_path):
            print("Downloading IMDb dataset...")
            response = requests.get(self.dataset_url, stream=True)
            total_size = int(response.headers.get("content-length", 0))
            with (
                open(archive_path, "wb") as f,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="aclImdb_v1.tar.gz",
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print("Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir)
        os.remove(archive_path)

    def load_raw_data(self):
        split_dir = os.path.join(self.data_dir, self.extract_name, self.split)
        pos_path = os.path.join(split_dir, "pos")
        neg_path = os.path.join(split_dir, "neg")

        texts, labels = [], []
        for label, folder in [(1, pos_path), (0, neg_path)]:
            for file in os.listdir(folder):
                with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                    text = f.read().strip().replace("\n", " ")
                    texts.append(text)
                    labels.append(label)
        return texts, labels

    def build_vocab(self, texts, min_freq=2):
        counter = Counter()
        for text in texts:
            tokens = text.lower().split()
            counter.update(tokens)

        vocab = {"<pad>": 0, "<unk>": 1}
        idx = 2
        for word, freq in counter.items():
            if freq >= min_freq:
                vocab[word] = idx
                idx += 1
        return vocab

    def encode_data(self, texts):
        encoded = []
        for text in texts:
            tokens = text.lower().split()
            indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
            encoded.append(indices)
        return encoded

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = self.encoded[idx][: self.seq_len]
        x = (
            x + [self.vocab["<pad>"]] * (self.seq_len - len(x))
            if len(x) < self.seq_len
            else x
        )
        return torch.tensor(x, dtype=torch.long), torch.tensor(
            self.labels[idx], dtype=torch.long
        )

    def get_vocab(self):
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def num_classes(self):
        return 2


def get_imdb_dataloaders(cfg):
    batch_size = cfg.dataset.batch_size

    train_dataset = IMDBDataset(cfg.dataset, split="train")
    vocab = train_dataset.get_vocab()
    test_dataset = IMDBDataset(cfg.dataset, split="test", vocab=vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return DatasetBundle(train_loader, test_loader, test_loader, vocab=vocab), vocab

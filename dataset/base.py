import os
import abc
import torch
import requests
from tqdm import tqdm
from collections import Counter
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any


class DatasetUtils:
    @staticmethod
    def download_file(url: str, filepath: str):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(filepath, "wb") as f,
            tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=os.path.basename(filepath),
            ) as pbar,
        ):
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    @staticmethod
    def ensure_dir(path: str):
        os.makedirs(path, exist_ok=True)


class BaseNLPDataset(Dataset, abc.ABC):
    """
    minimal abstract base class for NLP datasets.
    common functionality: vocab building, encoding, and padding.
    """

    def __init__(
        self, cfg: Dict[str, Any], split: str, vocab: Optional[Dict[str, int]] = None
    ):
        self.cfg = cfg
        self.data_dir: str = cfg["dataset"]["data_dir"]
        self.split = split
        self.seq_len: int = cfg["model"].get("seq_len", 512)
        self.min_freq: int = cfg["model"].get("min_freq", 2)

        self.prepare_data()
        self.texts, self.labels = self.load_raw_data()

        if vocab is None and split == "train":
            self.vocab = self.load_or_build_vocab(self.texts)
        elif vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError(
                "Vocabulary must be provided for non-train splits or build from train split first"
            )

        self.encoded = self.encode_data(self.texts)

    @abc.abstractmethod
    def prepare_data(self):
        """
        Prepare data (download, extract, etc.) if needed.
        Each dataset implements its own preparation logic.
        """
        pass

    @abc.abstractmethod
    def load_raw_data(self) -> Tuple[List[str], List[Any]]:
        pass

    @abc.abstractmethod
    def get_num_classes(self) -> int:
        pass

    def load_or_build_vocab(self, texts: List[str]) -> Dict[str, int]:
        if (
            self.cfg["model"].get("use_pretrained_embedding", False)
            and "vocab_path" in self.cfg["model"]
        ):
            print("Loading pretrained vocabulary...")
            pretrained_vocab = self.load_pretrained_vocab(
                self.cfg["model"]["vocab_path"]
            )
            vocab = self.ensure_special_tokens(pretrained_vocab)
            print(f"Loaded pretrained vocabulary with {len(vocab)} tokens")
            return vocab
        else:
            print("Building vocabulary from scratch...")
            vocab = self.build_vocab(texts)
            print(f"Built vocabulary with {len(vocab)} tokens")
            return vocab

    def load_pretrained_vocab(self, vocab_path: str) -> Dict[str, int]:
        try:
            from model.model_registry import load_vocab

            pretrained_vocab = load_vocab(vocab_path)
            return pretrained_vocab.get("word2idx", pretrained_vocab)
        except ImportError:
            raise NotImplementedError(
                "load_pretrained_vocab must be implemented or model.model_registry must be available"
            )

    def ensure_special_tokens(self, pretrained_vocab: Dict[str, int]) -> Dict[str, int]:
        """
        override to customize special token handling.
        """
        special_tokens = self.get_special_tokens()
        vocab = pretrained_vocab.copy()

        max_idx = max(vocab.values()) if vocab else -1
        next_idx = max_idx + 1

        for token_name, _ in special_tokens.items():
            if token_name not in vocab:
                vocab[token_name] = next_idx
                print(f"""Added special token '{token_name}' with index {next_idx}""")
                next_idx += 1

        return vocab

    def build_vocab(self, texts: List[str]) -> Dict[str, int]:
        """
        Override if you need custom vocab logic.
        """
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

    def get_special_tokens(self) -> Dict[str, int]:
        """
        Override if you need different special tokens.
        """
        return {"<pad>": 0, "<unk>": 1}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def encode_data(self, texts: List[str]) -> List[List[int]]:
        encoded = []
        for text in texts:
            tokens = self.tokenize(text)
            indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
            encoded.append(indices)
        return encoded

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        default implementation for classification tasks.
        override for other tasks like language modeling.
        """
        x = self.encoded[index][: self.seq_len]

        if len(x) < self.seq_len:
            x = x + [self.vocab["<pad>"]] * (self.seq_len - len(x))

        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
        )

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def num_classes(self) -> int:
        return self.get_num_classes()


class DatasetBundle:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        vocab=None,
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.vocab = vocab

    @property
    def vocab_size(self):
        return self.train_loader.dataset.vocab_size

    @property
    def num_classes(self):
        return self.train_loader.dataset.num_classes

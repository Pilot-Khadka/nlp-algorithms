import os
import requests
import zipfile
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.base import DatasetBundle
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
from collections import Counter


class SST2Dataset(Dataset):
    """
    PyTorch Dataset class for SST-2 (Stanford Sentiment Treebank) dataset.

    The SST-2 dataset contains movie reviews with binary sentiment labels:
    - 0: Negative sentiment
    - 1: Positive sentiment
    """

    def __init__(
        self,
        cfg,
        split: str = "train",
        vocab: Optional[Dict[str, int]] = None,
    ):
        self.data_dir = cfg["data_dir"]
        self.split = split
        self.seq_len = cfg.get("seq_len", 128)

        if not os.path.exists(self.data_dir):
            self.url = cfg["url"]
            print("File not found, downloading the dataset")
            self.download_data()

        self.sentences, self.labels = self._load_data(split)

        if vocab is None and split == "train":
            self.vocab = self.build_vocab(self.sentences)
            print(f"Built vocabulary with {len(self.vocab)} tokens")
        elif vocab is not None:
            self.vocab = vocab
        else:
            raise ValueError(
                "Vocabulary must be provided for non-train splits or build from train split first"
            )

        self.encoded_sentences = self.encode_sentences(self.sentences)
        print(f"Loaded {len(self.sentences)} samples from {split} split")

    def download_data(self):
        os.makedirs(self.data_dir, exist_ok=True)
        file_path = os.path.join(self.data_dir, "SST-2.zip")
        if not os.path.exists(file_path):
            print("Downloading SST-2 dataset...")
            response = requests.get(self.url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            #  'wb' for binary mode since it's a zip file
            with open(file_path, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="SST-2.zip"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

        print("Extracting SST-2 dataset...")
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        os.remove(file_path)

        files_found = []
        if os.path.exists(self.data_dir):
            all_files = os.listdir(self.data_dir)
            files_found = [
                file
                for file in all_files
                if any(
                    pattern in file.lower()
                    for pattern in ["train", "dev", "valid", "test"]
                )
            ]

        # if files are not found directly, look in subdirectories
        if not files_found:
            for item in os.listdir(self.data_dir):
                item_path = os.path.join(self.data_dir, item)
                if os.path.isdir(item_path):
                    sub_files = os.listdir(item_path)
                    sub_files_found = [
                        file
                        for file in sub_files
                        if any(
                            pattern in file.lower()
                            for pattern in ["train", "dev", "valid", "test"]
                        )
                    ]

                    if sub_files_found:
                        print(f"Found files in subdirectory: {item}")
                        # move files from subdirectory to parent directory
                        for file in sub_files_found:
                            src = os.path.join(item_path, file)
                            dst = os.path.join(self.data_dir, file)
                            os.rename(src, dst)
                            print(f"Moved {file} to main directory")

                        # remove the now-empty subdirectory
                        try:
                            os.rmdir(item_path)
                            print(f"Removed empty directory: {item}")
                        except OSError:
                            # directory not empty, remove recursively
                            import shutil

                            shutil.rmtree(item_path)
                            print(f"Removed directory: {item}")
                        break

        final_files = os.listdir(self.data_dir)
        dataset_files = [
            file
            for file in final_files
            if any(
                pattern in file.lower() for pattern in ["train", "dev", "valid", "test"]
            )
        ]
        print(f"Dataset files ready: {dataset_files}")

    def _load_data(self, split) -> Tuple[List[str], List[int]]:
        sentences = []
        labels = []

        possible_files = []
        if split == "valid":
            possible_files = [f"dev.txt", f"valid.txt"]
        else:
            possible_files = [f"{split}.txt"]

        file_path = None
        for filename in possible_files:
            candidate_path = os.path.join(self.data_dir, filename)
            if os.path.exists(candidate_path):
                file_path = candidate_path
                break

        if file_path is None:
            raise FileNotFoundError(
                f"""No data file found for split '{
                    split}'. Checked: {possible_files}"""
            )

        print(f"Loading data from: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if line:
                try:
                    sentence, label = self._parse_tree(line)
                    sentences.append(sentence)
                    labels.append(label)
                except Exception as e:
                    print(f"Error parsing line: {line[:50]}... Error: {e}")
                    continue

        return sentences, labels

    def _parse_tree(self, tree_string: str) -> Tuple[str, int]:
        tree_string = tree_string.strip()

        if not tree_string.startswith("("):
            raise ValueError(f"Invalid tree format: {tree_string}")

        root_label = None
        i = 1  # skip opening parenthesis
        while i < len(tree_string) and tree_string[i].isdigit():
            if root_label is None:
                root_label = int(tree_string[i])
            i += 1

        if root_label is None:
            raise ValueError(f"Could not find root label in: {tree_string}")

        # Convert 5-class labels to binary (SST-2)
        # Original: 0=very negative, 1=negative, 2=neutral, 3=positive, 4=very positive
        # Binary: 0-2 -> 0 (negative), 3-4 -> 1 (positive)
        binary_label = 1 if root_label >= 3 else 0

        words = self._extract_words(tree_string)
        sentence = " ".join(words)

        return sentence, binary_label

    def _extract_words(self, tree_string: str) -> List[str]:
        words = []
        i = 0
        while i < len(tree_string):
            if tree_string[i] == "(":
                # skip opening parenthesis and label
                i += 1
                # skip label (digit)
                while i < len(tree_string) and (
                    tree_string[i].isdigit() or tree_string[i].isspace()
                ):
                    i += 1
            elif tree_string[i] == ")":
                # skipj closing parenthesis
                i += 1
            elif tree_string[i].isspace():
                # skip whitespace
                i += 1
            else:
                # extract word
                word_start = i
                while i < len(tree_string) and tree_string[i] not in "() ":
                    i += 1
                word = tree_string[word_start:i]
                if word:  # Only add non-empty words
                    words.append(word)

        return words

    def build_vocab(self, sentences: List[str], min_freq: int = 1) -> Dict[str, int]:
        all_tokens = []
        for sentence in sentences:
            tokens = sentence.lower().split()
            all_tokens.extend(tokens)

        counter = Counter(all_tokens)

        vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2}
        idx = 3

        for token, freq in counter.most_common():
            if freq >= min_freq and token not in vocab:
                vocab[token] = idx
                idx += 1

        return vocab

    def encode_sentences(self, sentences: List[str]) -> List[List[int]]:
        encoded = []
        for sentence in sentences:
            tokens = sentence.lower().split()
            if len(tokens) > self.seq_len - 1:  # -1 for EOS token
                tokens = tokens[: self.seq_len - 1]

            indices = [self.vocab.get(token, self.vocab["<unk>"])
                       for token in tokens]
            indices.append(self.vocab["<eos>"])

            while len(indices) < self.seq_len:
                indices.append(self.vocab["<pad>"])

            encoded.append(indices)

        return encoded

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def num_classes(self) -> int:
        return 2  # SST-2 is binary classification (negative/positive)

    def get_num_classes(self) -> int:
        return 2

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_sentence = torch.tensor(
            self.encoded_sentences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return encoded_sentence, label

    def get_label_distribution(self) -> Dict[str, int]:
        if -1 in self.labels:
            return {"unknown": len(self.labels)}

        unique_labels, counts = torch.unique(
            torch.tensor(self.labels), return_counts=True
        )
        return {
            f"label_{label.item()}": count.item()
            for label, count in zip(unique_labels, counts)
        }


def get_sst2_dataloaders(cfg):
    batch_size = cfg.batch_size

    train_dataset = SST2Dataset(cfg, "train")
    vocab = train_dataset.get_vocab()

    valid_dataset = SST2Dataset(cfg, "valid", vocab=vocab)
    test_dataset = SST2Dataset(cfg, "test", vocab=vocab)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return DatasetBundle(
        train_loader,
        valid_loader,
        test_loader,
    )


import os
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader

from datasets.base import DatasetBundle, DatasetUtils, BaseNLPDataset


class SST2Dataset(BaseNLPDataset):
    def prepare_data(self):
        if not os.path.exists(self.data_dir):
            self._download_and_extract()

    def _download_and_extract(self):
        import zipfile

        zip_name = "SST-2.zip"
        temp_zip = zip_name + ".tmp"

        try:
            print("Downloading SST-2 dataset...")
            DatasetUtils.download_file(self.cfg["dataset"]["url"], temp_zip)

            DatasetUtils.ensure_dir(self.data_dir)
            final_zip = os.path.join(self.data_dir, zip_name)
            os.rename(temp_zip, final_zip)

            print("Extracting SST-2 dataset...")
            with zipfile.ZipFile(final_zip, "r") as zip_ref:
                zip_ref.extractall(self.data_dir)

            os.remove(final_zip)
            self._organize_files()

        except (KeyboardInterrupt, Exception):
            if os.path.exists(temp_zip):
                os.remove(temp_zip)
            raise

    def _organize_files(self):
        files_found = [
            f
            for f in os.listdir(self.data_dir)
            if any(
                pattern in f.lower() for pattern in ["train", "dev", "valid", "test"]
            )
        ]

        if not files_found:
            for item in os.listdir(self.data_dir):
                item_path = os.path.join(self.data_dir, item)
                if os.path.isdir(item_path):
                    sub_files = [
                        f
                        for f in os.listdir(item_path)
                        if any(
                            pattern in f.lower()
                            for pattern in ["train", "dev", "valid", "test"]
                        )
                    ]

                    if sub_files:
                        for file in sub_files:
                            src = os.path.join(item_path, file)
                            dst = os.path.join(self.data_dir, file)
                            os.rename(src, dst)

                        import shutil

                        shutil.rmtree(item_path)
                        break

    def ensure_special_tokens(self, pretrained_vocab: Dict[str, int]) -> Dict[str, int]:
        special_tokens = ["<pad>", "<unk>", "<eos>"]
        vocab = pretrained_vocab.copy()

        max_idx = max(vocab.values()) if vocab else -1
        next_idx = max_idx + 1

        for token in special_tokens:
            if token not in vocab:
                vocab[token] = next_idx
                print(f"Added special token '{token}' with index {next_idx}")
                next_idx += 1

        return vocab

    def get_special_tokens(self) -> Dict[str, int]:
        return {"<pad>": 0, "<unk>": 1, "<eos>": 2}

    def load_raw_data(self) -> Tuple[List[str], List[int]]:
        """Load SST2 data from tree format."""
        possible_files = (
            ["dev.txt", "valid.txt"] if self.split == "valid" else [f"{self.split}.txt"]
        )

        file_path = None
        for filename in possible_files:
            candidate_path = os.path.join(self.data_dir, filename)
            if os.path.exists(candidate_path):
                file_path = candidate_path
                break

        if file_path is None:
            raise FileNotFoundError(
                f"""No data file found for split '{self.split}'. Checked: {
                    possible_files
                }"""
            )

        sentences, labels = [], []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
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
        i = 1
        while i < len(tree_string) and tree_string[i].isdigit():
            if root_label is None:
                root_label = int(tree_string[i])
            i += 1

        if root_label is None:
            raise ValueError(f"Could not find root label in: {tree_string}")

        # convert to binary (0-2 -> 0, 3-4 -> 1)
        binary_label = 1 if root_label >= 3 else 0

        words = self._extract_words(tree_string)
        sentence = " ".join(words)

        return sentence, binary_label

    def _extract_words(self, tree_string: str) -> List[str]:
        words = []
        i = 0
        while i < len(tree_string):
            if tree_string[i] == "(":
                i += 1
                while i < len(tree_string) and (
                    tree_string[i].isdigit() or tree_string[i].isspace()
                ):
                    i += 1
            elif tree_string[i] == ")":
                i += 1
            elif tree_string[i].isspace():
                i += 1
            else:
                word_start = i
                while i < len(tree_string) and tree_string[i] not in "() ":
                    i += 1
                word = tree_string[word_start:i]
                if word:
                    words.append(word)
        return words

    def encode_data(self, texts: List[str]) -> List[List[int]]:
        encoded = []
        for text in texts:
            tokens = self.tokenize(text)
            if len(tokens) > self.seq_len - 1:  # -1 for EOS token
                tokens = tokens[: self.seq_len - 1]

            indices = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
            indices.append(self.vocab["<eos>"])

            while len(indices) < self.seq_len:
                indices.append(self.vocab["<pad>"])

            encoded.append(indices)
        return encoded

    def get_num_classes(self) -> int:
        return 2


def get_sst2_dataloaders(cfg):
    batch_size = cfg.dataset.batch_size

    train_dataset = SST2Dataset(cfg, "train", vocab=None)
    final_vocab = train_dataset.get_vocab()

    valid_dataset = SST2Dataset(cfg, "valid", vocab=final_vocab)
    test_dataset = SST2Dataset(cfg, "test", vocab=final_vocab)

    return DatasetBundle(
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(valid_dataset, batch_size=batch_size),
        DataLoader(test_dataset, batch_size=batch_size),
        vocab=final_vocab,
    )

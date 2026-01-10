import os
import shutil
import tarfile
from torch.utils.data import DataLoader, Dataset
from typing import Dict, Any, Optional, Tuple, List

from datasets.base import DatasetBundle, DatasetUtils


class TatoebaDataset(Dataset):
    def __init__(
        self,
        cfg: Dict[str, Any],
        split: str,
        source_vocab: Optional[Dict[str, int]] = None,
        target_vocab: Optional[Dict[str, int]] = None,
    ):
        self.dataset_url = cfg["dataset"]["url"]
        self.archive_name = os.path.basename(cfg["dataset"]["url"])
        self.extract_name = self.archive_name.rsplit(".tar", 1)[0]
        self.data_dir = cfg["dataset"]["data_dir"]
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        self.split = split

    def prepare_data(self):
        extract_path = os.path.join(self.data_dir, self.extract_name)

        if os.path.exists(extract_path):
            return

        if not os.path.exists(self.data_dir):
            self._download_and_extract()

    def _download_and_extract(self):
        archive_filename = self.archive_name
        extract_name = self.extract_name

        temp_archive = archive_filename + ".tmp"
        temp_extract_path = os.path.join(self.data_dir, extract_name + ".tmp")
        extract_path = os.path.join(self.data_dir, extract_name)

        try:
            if not os.path.exists(os.path.join(self.data_dir, archive_filename)):
                print("Downloading Tatoeba dataset...")
                DatasetUtils.download_file(self.dataset_url, temp_archive)

                DatasetUtils.ensure_dir(self.data_dir)
                final_archive = os.path.join(self.data_dir, archive_filename)
                os.rename(temp_archive, final_archive)
            else:
                final_archive = os.path.join(self.data_dir, archive_filename)

            print("Extracting...")
            DatasetUtils.ensure_dir(self.data_dir)

            with tarfile.open(final_archive, "r") as tar:
                tar.extractall(path=temp_extract_path, filter="data")

            os.rename(temp_extract_path, extract_path)
            os.remove(final_archive)

        except (KeyboardInterrupt, Exception):
            if os.path.exists(temp_archive):
                os.remove(temp_archive)
            if os.path.exists(temp_extract_path):
                shutil.rmtree(temp_extract_path, ignore_errors=True)
            if os.path.exists(extract_path) and not self._is_complete_extraction(
                extract_path
            ):
                shutil.rmtree(extract_path, ignore_errors=True)
            raise

    def _is_complete_extraction(self, path: str) -> bool:
        for split in ["train", "dev", "test"]:
            src_file = os.path.join(path, f"{split}.src")
            trg_file = os.path.join(path, f"{split}.trg")
            if not os.path.exists(src_file) or not os.path.exists(trg_file):
                return False
        return True

    def load_raw_data(self) -> Tuple[List[str], List[str]]:
        dataset_path = os.path.join(self.data_dir, self.extract_name)
        src_file = os.path.join(dataset_path, f"{self.split}.src")
        trg_file = os.path.join(dataset_path, f"{self.split}.trg")

        source_texts, target_texts = [], []

        with (
            open(src_file, "r", encoding="utf-8") as src_f,
            open(trg_file, "r", encoding="utf-8") as trg_f,
        ):
            for src_line, trg_line in zip(src_f, trg_f):
                source_texts.append(src_line.strip())
                target_texts.append(trg_line.strip())

        return source_texts, target_texts


def get_tatoeba_dataloaders(cfg):
    print("cfg training:", cfg.training)
    batch_size = cfg.training.batch_size
    train_dataset = TatoebaDataset(cfg, split="train")
    # pyrefly: ignore [missing-attribute]
    source_vocab = train_dataset.get_vocab()

    train_texts_src, train_texts_trg = train_dataset.load_raw_data()
    # pyrefly: ignore [missing-attribute]
    target_vocab = DatasetUtils.build_vocab(train_texts_trg, cfg.dataset.vocab_size)

    train_dataset = TatoebaDataset(
        cfg, split="train", source_vocab=source_vocab, target_vocab=target_vocab
    )
    dev_dataset = TatoebaDataset(
        cfg, split="dev", source_vocab=source_vocab, target_vocab=target_vocab
    )
    test_dataset = TatoebaDataset(
        cfg, split="test", source_vocab=source_vocab, target_vocab=target_vocab
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return DatasetBundle(
        train_loader,
        dev_loader,
        test_loader,
        vocab=source_vocab,
        # pyrefly: ignore [unexpected-keyword]
        target_vocab=target_vocab,
    )

import os
import shutil
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Tuple, List


from datasets.base import DatasetBundle
from datasets.base import DatasetUtils, BaseNLPDataset


class IMDBDataset(BaseNLPDataset):
    def __init__(
        self, cfg: Dict[str, Any], split: str, vocab: Optional[Dict[str, int]] = None
    ):
        self.dataset_url = cfg["dataset"]["url"]
        self.archive_name = cfg["dataset"]["archive_name"]
        self.extract_name = self.archive_name.split(".")[0].replace("_v1", "")
        super().__init__(cfg, split, vocab)

    def prepare_data(self):
        extract_path = os.path.join(self.data_dir, self.extract_name)

        if os.path.exists(extract_path):
            return

        if not os.path.exists(self.data_dir):
            self._download_and_extract()

    def _download_and_extract(self):
        import tarfile

        # construct filenames only
        archive_filename = self.archive_name
        extract_name = self.extract_name

        # temp files stored in working directory to avoid nonexistent parent dirs
        temp_archive = archive_filename + ".tmp"

        try:
            if not os.path.exists(os.path.join(self.data_dir, archive_filename)):
                print("Downloading IMDb dataset...")
                DatasetUtils.download_file(self.dataset_url, temp_archive)

                DatasetUtils.ensure_dir(self.data_dir)
                final_archive = os.path.join(self.data_dir, archive_filename)
                os.rename(temp_archive, final_archive)
            else:
                final_archive = os.path.join(self.data_dir, archive_filename)

            print("Extracting...")
            extract_path = os.path.join(self.data_dir, extract_name)
            temp_extract_path = extract_path + ".tmp"

            with tarfile.open(final_archive, "r:gz") as tar:
                tar.extractall(path=self.data_dir, filter="data")

            os.rename(extract_path, temp_extract_path)
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
        """Check if extraction completed by verifying expected structure exists"""
        for split in ["train", "test"]:
            split_dir = os.path.join(path, split)
            if not os.path.exists(split_dir):
                return False
            for label_dir in ["pos", "neg"]:
                if not os.path.exists(os.path.join(split_dir, label_dir)):
                    return False
        return True

    def load_raw_data(self) -> Tuple[List[str], List[int]]:
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

    def get_num_classes(self) -> int:
        return 2


def get_imdb_dataloaders(cfg):
    batch_size = cfg.dataset.batch_size
    train_dataset = IMDBDataset(cfg, split="train")
    vocab = train_dataset.get_vocab()
    test_dataset = IMDBDataset(cfg, split="test", vocab=vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return DatasetBundle(train_loader, test_loader, test_loader, vocab=vocab)

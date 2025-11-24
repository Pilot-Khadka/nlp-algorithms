import os
from typing import Dict, Any, Optional, Tuple, List

from torch.utils.data import DataLoader
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
        if not os.path.exists(self.data_dir):
            self._download_and_extract()

    def _download_and_extract(self):
        import tarfile

        DatasetUtils.ensure_dir(self.data_dir)
        archive_path = os.path.join(self.data_dir, self.archive_name)

        if not os.path.exists(archive_path):
            print("Downloading IMDb dataset...")
            DatasetUtils.download_file(self.dataset_url, archive_path)

        print("Extracting...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=self.data_dir)
        os.remove(archive_path)

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

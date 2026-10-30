from typing import Dict, Any


import os
import glob
import threading
from filelock import FileLock
import torch.distributed as dist

from nlp_algorithms.util.multi_gpu import is_rank0
from nlp_algorithms.dataset.downloader.huggingface_downloader import (
    HuggingFaceDatasetDownloader,
)

from nlp_algorithms.engine.registry import register_downloader


@register_downloader("imdb")
class ImdbDownloader:
    _thread_lock = threading.Lock()

    @classmethod
    def download_and_prepare(
        cls,
        cfg: Dict[str, Any],
        force: bool = False,
    ):
        # the same 25k/25k reviews as the stanford archive, as parquet on hugging face
        data_dir = cfg["dataset"]["data_dir"]
        repo_id = cfg["dataset"].get("repo", "stanfordnlp/imdb")
        os.makedirs(data_dir, exist_ok=True)

        extract_path = os.path.join(data_dir, "plain_text")
        prepared_flag = os.path.join(extract_path, ".prepared")
        lock_path = os.path.join(data_dir, ".lock")

        # train_split "train+unsupervised" adds the 50k unlabeled reviews (pretraining)
        splits = sorted(
            set(cfg["dataset"].get("train_split", "train").split("+")) | {"test"}
        )

        def prepared():
            return os.path.exists(prepared_flag) and all(
                glob.glob(os.path.join(extract_path, f"{split}-*.parquet"))
                for split in splits
            )

        if prepared() and not force:
            return extract_path

        with cls._thread_lock:
            with FileLock(lock_path):
                if prepared() and not force:
                    return extract_path

                if is_rank0():
                    HuggingFaceDatasetDownloader._download_repo(
                        repo_id,
                        data_dir,
                        [f"plain_text/{split}-*.parquet" for split in splits],
                    )

                    with open(prepared_flag, "w") as f:
                        f.write("ok")

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        return extract_path

from typing import Dict, Any


import os
import tarfile
import shutil
import threading
from filelock import FileLock
import torch.distributed as dist
from urllib.parse import urlparse

from util.multi_gpu import is_rank0
from util.data import download_file


class ImdbDownloader:
    _thread_lock = threading.Lock()

    @classmethod
    def download_and_prepare(
        cls,
        cfg: Dict[str, Any],
        force: bool = False,
    ):
        data_dir = cfg["dataset"]["data_dir"]
        dataset_url = cfg["dataset"]["url"]
        os.makedirs(data_dir, exist_ok=True)

        archive_name = os.path.basename(urlparse(dataset_url).path)
        archive_path = os.path.join(data_dir, archive_name)

        extract_name = archive_name.split(".")[0].replace("_v1", "")
        extract_path = os.path.join(data_dir, extract_name)

        prepared_flag = os.path.join(extract_path, ".prepared")
        lock_path = os.path.join(data_dir, ".lock")

        if os.path.exists(prepared_flag) and not force:
            return extract_path

        with cls._thread_lock:
            with FileLock(lock_path):
                if os.path.exists(prepared_flag) and not force:
                    return extract_path

                if is_rank0():
                    print(f"[IMDb] Rank0 writing prepared flag at {prepared_flag}")
                    if not os.path.exists(archive_path) or force:
                        print(f"[IMDb] Downloading to {archive_path} ...")
                        temp_path = archive_path + ".tmp"
                        download_file(url=dataset_url, filepath=temp_path)
                        os.replace(temp_path, archive_path)

                    print(f"[IMDb] Extracting to {extract_path} ...")
                    cls._safe_extract(archive_path, data_dir, extract_path)

                    with open(prepared_flag, "w") as f:
                        f.write("ok")

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

        return extract_path

    @staticmethod
    def _safe_extract(archive_path, data_dir, extract_path):
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

        if os.path.exists(archive_path):
            os.remove(archive_path)

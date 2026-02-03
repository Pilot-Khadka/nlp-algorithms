from typing import Any, Dict

import os
import shutil
import threading
from tqdm import tqdm
from filelock import FileLock

import torch.distributed as dist
from huggingface_hub import list_repo_files, hf_hub_download

from util.multi_gpu import is_rank0
from engine.registry import register_downloader


@register_downloader("huggingface_dataset")
class HuggingFaceDatasetDownloader:
    _thread_lock = threading.Lock()

    @classmethod
    def download_and_prepare(
        cls,
        cfg: Dict[str, Any],
        force: bool = False,
    ) -> str:
        data_dir = cfg["dataset"]["data_dir"]
        repo_id = cfg["dataset"]["repo_id"]
        os.makedirs(data_dir, exist_ok=True)

        prepared_flag = os.path.join(data_dir, ".prepared")
        lock_path = os.path.join(data_dir, ".lock")

        # quick-return
        if os.path.exists(prepared_flag) and not force:
            return data_dir

        with cls._thread_lock:
            with FileLock(lock_path):
                if os.path.exists(prepared_flag) and not force:
                    return data_dir

                if is_rank0():
                    print(
                        f"[Rank 0] Downloading & preparing dataset from HF repo: {repo_id}"
                    )

                    try:
                        cls._download_all_files(repo_id, data_dir)

                        with open(prepared_flag, "w") as f:
                            f.write("ok")

                        print("Dataset successfully prepared.")

                    except Exception as e:
                        if os.path.exists(prepared_flag):
                            os.remove(prepared_flag)
                        raise e

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

                if not os.path.exists(prepared_flag):
                    raise RuntimeError(
                        "Dataset preparation failed (no .prepared created)"
                    )

        return data_dir

    @staticmethod
    def _download_all_files(repo_id: str, data_dir: str):
        print("Listing HuggingFace repo files...")
        files = list_repo_files(repo_id, repo_type="dataset")

        for filename in tqdm(files, desc="Downloading files from HF"):
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="dataset"
                )
                dst_path = os.path.join(data_dir, filename)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(downloaded_path, dst_path)

            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                raise


if __name__ == "__main__":
    downloader = HuggingFaceDatasetDownloader()
    config = {
        "dataset": {
            "name": "eng-nep-parallel",
            "data_dir": "../dataset_hf_eng_nep",
            "repo_id": "sharad461/ne-en-parallel-208k",
        }
    }
    print("downloading")
    downloader.download_and_prepare(config)

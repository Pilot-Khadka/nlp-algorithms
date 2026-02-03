from typing import Any, Dict, List, Union

import os
import shutil
import fnmatch
import threading
from tqdm import tqdm
from filelock import FileLock

import torch.distributed as dist
from huggingface_hub import list_repo_files, hf_hub_download, login, whoami

from util.multi_gpu import is_rank0
from engine.registry import register_downloader


@register_downloader("huggingface")
class HuggingFaceDatasetDownloader:
    _thread_lock = threading.Lock()
    _logged_in = False

    @staticmethod
    def _ensure_hf_login():
        if HuggingFaceDatasetDownloader._logged_in:
            return

        token = os.environ.get("HF_TOKEN", None)

        if token is None:
            print(
                "[WARNING] HF_TOKEN not found in environment. "
                "If the repo is private, download will fail."
            )
            return

        try:
            login(token=token, add_to_git_credential=True)
            user = whoami()
            print(f"[HF Login] Logged in as: {user.get('name')}")
        except Exception as e:
            print(f"[ERROR] Failed to login to Hugging Face: {e}")
            raise

        HuggingFaceDatasetDownloader._logged_in = True

    @classmethod
    def download_and_prepare(
        cls,
        cfg: Dict[str, Any],
        force: bool = False,
    ) -> str:
        dataset_cfg = cfg["dataset"]
        data_dir = dataset_cfg["data_dir"]
        repos = dataset_cfg["repos"]

        os.makedirs(data_dir, exist_ok=True)

        prepared_flag = os.path.join(data_dir, ".prepared")
        lock_path = os.path.join(data_dir, ".lock")

        if os.path.exists(prepared_flag) and not force:
            return data_dir

        with cls._thread_lock:
            with FileLock(lock_path):
                if os.path.exists(prepared_flag) and not force:
                    return data_dir

                if is_rank0():
                    print("[Rank 0] Preparing HF datasets...")

                try:
                    for repo_cfg in repos:
                        repo_id = repo_cfg["id"]
                        patterns = repo_cfg.get("files", "*")

                        if is_rank0():
                            print(f"[Rank 0] Repo: {repo_id}, patterns={patterns}")

                        cls._download_repo(repo_id, data_dir, patterns)

                    with open(prepared_flag, "w") as f:
                        f.write("ok")

                    if is_rank0():
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
    def _filter_files(files: List[str], patterns: Union[str, List[str]]) -> List[str]:
        if patterns is None or patterns == "*" or patterns == ["*"]:
            return files

        if isinstance(patterns, str):
            patterns = [patterns]

        matched = []
        for pat in patterns:
            matched.extend(fnmatch.filter(files, pat))

        return list(sorted(set(matched)))

    @staticmethod
    def _download_repo(repo_id: str, data_dir: str, patterns):
        print(f"Listing files in HF repo: {repo_id}")
        all_files = list_repo_files(repo_id, repo_type="dataset")

        files_to_download = HuggingFaceDatasetDownloader._filter_files(
            all_files, patterns
        )

        print(
            f"Matched {len(files_to_download)} / {len(all_files)} files "
            f"in repo {repo_id} using patterns {patterns}"
        )

        for filename in tqdm(files_to_download, desc=f"Downloading from {repo_id}"):
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
            "name": "huggingface",
            "data_dir": "../dataset_hf_eng_nep",
            "repos": [
                {
                    "id": "sharad461/ne-en-parallel-208k",
                    "files": "*",  # or omit "files" to download all
                },
                {
                    "id": "openlanguagedata/flores_plus",
                    "files": [
                        # Nepali (Devanagari)
                        "dev/npi_Deva.jsonl",
                        "devtest/npi_Deva.jsonl",
                        # English (Latin)
                        "dev/eng_Latn.jsonl",
                        "devtest/eng_Latn.jsonl",
                    ],
                },
            ],
        }
    }

    downloader.download_and_prepare(config)

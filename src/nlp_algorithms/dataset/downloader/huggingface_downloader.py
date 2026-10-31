from typing import Any, Dict, List, Union

import os
import time
import shutil
import logging
import fnmatch
import threading
from filelock import FileLock

import torch.distributed as dist
from huggingface_hub import list_repo_files, hf_hub_download, login, whoami

from nlp_algorithms.util.logger import fmt_time
from nlp_algorithms.engine.registry import register_downloader

logger = logging.getLogger(__name__)


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
            logger.warning("HF_TOKEN is not set, private repos will fail to download")
            return

        try:
            login(token=token)
            user = whoami()
            logger.info(f"hugging face login as {user.get('name')}")
        except Exception as e:
            logger.error(f"hugging face login failed: {e}")
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

                try:
                    for repo_cfg in repos:
                        repo_id = repo_cfg["id"]
                        patterns = repo_cfg.get("files", "*")
                        cls._download_repo(repo_id, data_dir, patterns)

                    with open(prepared_flag, "w") as f:
                        f.write("ok")

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
        all_files = list_repo_files(repo_id, repo_type="dataset")

        files_to_download = HuggingFaceDatasetDownloader._filter_files(
            all_files, patterns
        )

        logger.info(
            f"{repo_id}: downloading {len(files_to_download)} of {len(all_files)} files "
            f"(patterns {patterns}) to {data_dir}"
        )

        start = time.perf_counter()
        size = 0
        for i, filename in enumerate(files_to_download, 1):
            try:
                downloaded_path = hf_hub_download(
                    repo_id=repo_id, filename=filename, repo_type="dataset"
                )
                dst_path = os.path.join(data_dir, filename)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copy(downloaded_path, dst_path)
                size += os.path.getsize(dst_path)
                logger.info(
                    f"{repo_id}: {i}/{len(files_to_download)} {filename}, "
                    f"{size / 1e6:.1f} MB so far, {fmt_time(time.perf_counter() - start)}"
                )

            except Exception as e:
                logger.error(f"{repo_id}: failed to download {filename}: {e}")
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

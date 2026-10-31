from typing import Any, Dict

import os
import time
import gzip
import logging
import shutil
import tarfile
import threading
from filelock import FileLock

import torch.distributed as dist

from nlp_algorithms.util.data import download_file
from nlp_algorithms.util.logger import fmt_time
from nlp_algorithms.util.multi_gpu import is_rank0
from nlp_algorithms.engine.registry import register_downloader

logger = logging.getLogger(__name__)


@register_downloader("tatoeba")
class TatoebaDownloader:
    _thread_lock = threading.Lock()

    @classmethod
    def download_and_prepare(
        cls,
        cfg: Dict[str, Any],
        force: bool = False,
    ) -> str:
        data_dir = cfg["dataset"]["data_dir"]
        dataset_url = cfg["dataset"]["url"]
        os.makedirs(data_dir, exist_ok=True)

        prepared_flag = os.path.join(data_dir, ".prepared")
        lock_path = os.path.join(data_dir, ".lock")

        if os.path.exists(prepared_flag) and not force:
            return data_dir

        with cls._thread_lock:
            with FileLock(lock_path):
                # re-check because another process may have finished while we waited
                if os.path.exists(prepared_flag) and not force:
                    return data_dir

                if is_rank0():
                    try:
                        cls._download_and_extract_internal(dataset_url, data_dir)
                        cls._validate_required_files(data_dir)

                        with open(prepared_flag, "w") as f:
                            f.write("ok")

                    except Exception as e:
                        # remove flag if incomplete
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
    def _download_and_extract_internal(dataset_url: str, data_dir: str):
        archive_name = os.path.basename(dataset_url)
        archive_path = os.path.join(data_dir, archive_name)
        temp_archive = archive_path + ".tmp"
        temp_extract_dir = os.path.join(data_dir, "tmp_extract")

        try:
            if not os.path.exists(archive_path):
                download_file(dataset_url, temp_archive)
                os.rename(temp_archive, archive_path)

            # the .gz files inside are decompressed too, a few minutes for large pairs
            logger.info(f"extracting {archive_path} to {data_dir}")
            start = time.perf_counter()
            os.makedirs(temp_extract_dir, exist_ok=True)
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=temp_extract_dir)

            TatoebaDownloader._flatten_extracted_files(temp_extract_dir, data_dir)
            shutil.rmtree(temp_extract_dir, ignore_errors=True)
            logger.info(
                f"extracted {sorted(os.listdir(data_dir))} in "
                f"{fmt_time(time.perf_counter() - start)}"
            )

        except Exception as e:
            logger.error(f"extraction of {archive_path} failed: {e}")
            if os.path.exists(temp_archive):
                os.remove(temp_archive)
            if os.path.exists(temp_extract_dir):
                shutil.rmtree(temp_extract_dir, ignore_errors=True)
            raise

    @staticmethod
    def _flatten_extracted_files(src_root: str, dst_root: str):
        os.makedirs(dst_root, exist_ok=True)

        for root, _, files in os.walk(src_root):
            for file in files:
                src_path = os.path.join(root, file)

                if os.path.abspath(root) == os.path.abspath(dst_root):
                    continue

                dst_name = file
                dst_path = os.path.join(dst_root, dst_name)

                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(dst_name)
                    count = 1
                    while os.path.exists(dst_path):
                        dst_name = f"{base}_{count}{ext}"
                        dst_path = os.path.join(dst_root, dst_name)
                        count += 1

                if file.endswith(".gz"):
                    decompressed_path = os.path.join(dst_root, dst_name[:-3])
                    with gzip.open(src_path, "rt", encoding="utf-8") as f_in:
                        with open(decompressed_path, "w", encoding="utf-8") as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(src_path)
                else:
                    shutil.move(src_path, dst_path)

        for root, dirs, _ in os.walk(src_root, topdown=False):
            for d in dirs:
                dir_path = os.path.join(root, d)
                try:
                    os.rmdir(dir_path)
                except OSError:
                    pass

    @staticmethod
    def _validate_required_files(data_dir: str):
        required_files = {"train.src", "train.trg", "test.src", "test.trg"}
        existing = set(os.listdir(data_dir))
        missing = required_files - existing

        if missing:
            raise RuntimeError(f"Dataset preparation incomplete, missing: {missing}")

import os
import threading

import torch.distributed as dist
from filelock import FileLock

from util.data import download_file
from util.multi_gpu import is_rank0

from engine.registry import register_downloader


@register_downloader("ptb")
class PTBDownloader:
    _thread_lock = threading.Lock()

    @classmethod
    def download_and_prepare(
        cls,
        cfg,
        force=False,
    ):
        ds_cfg = cfg["dataset"]
        data_dir = ds_cfg["data_dir"]
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
                    for key, value in ds_cfg.items():
                        if "url" in key:
                            filename = value.split("/")[-1]
                            dest_path = os.path.join(data_dir, filename)

                            print(f"Downloading {key} to {dest_path}...")
                            download_file(url=value, filepath=dest_path)

                    with open(prepared_flag, "w") as f:
                        f.write("ok")

                if dist.is_initialized():
                    dist.barrier()

        return data_dir

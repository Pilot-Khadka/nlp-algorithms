import os
import sys
import time
import logging

import requests
from tqdm import tqdm

from nlp_algorithms.util.logger import fmt_time

logger = logging.getLogger(__name__)


def download_file(url: str, filepath: str):
    response = requests.get(url, stream=True, timeout=60)
    # an error page would be saved as the archive and fail later in tarfile
    response.raise_for_status()
    # content-length is the compressed size when the server gzips the response
    total_size = 0
    if not response.headers.get("content-encoding"):
        total_size = int(response.headers.get("content-length", 0))
    size_text = f"{total_size / 1e6:.1f} MB" if total_size else "unknown size"
    logger.info(f"downloading {url} ({size_text}) to {filepath}")

    # the bar only in a terminal, piped logs (kaggle) get a line every 30s instead
    show_bar = sys.stderr.isatty()
    start = last_log = time.perf_counter()
    done = 0
    with (
        open(filepath, "wb") as f,
        tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=os.path.basename(filepath),
            disable=not show_bar,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=1 << 20):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
                done += len(chunk)
                now = time.perf_counter()
                if not show_bar and now - last_log >= 30:
                    last_log = now
                    pct = f" ({100 * done / total_size:.0f}%)" if total_size else ""
                    logger.info(f"downloaded {done / 1e6:.1f} MB{pct}")

    elapsed = time.perf_counter() - start
    logger.info(
        f"downloaded {done / 1e6:.1f} MB in {fmt_time(elapsed)} "
        f"({done / 1e6 / max(elapsed, 1e-6):.1f} MB/s)"
    )

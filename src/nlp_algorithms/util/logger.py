import os
import logging


def setup_logging(log_dir="log"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # hugging face downloads log every http request at info
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # prevent adding handlers multiple times
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # decide where logs go
        file_handler = logging.FileHandler(os.path.join(log_dir, "training.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # for real time feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def fmt_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"

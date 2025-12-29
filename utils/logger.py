import os
import logging


def setup_logging(log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

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

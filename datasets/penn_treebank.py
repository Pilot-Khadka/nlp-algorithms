import os
import requests

DATA_DIR = "./ptb_data"
URLS = {
    "train": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
    "valid": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
    "test": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
}

os.makedirs(DATA_DIR, exist_ok=True)


def download_ptb():
    for split, url in URLS.items():
        file_path = os.path.join(DATA_DIR, f"{split}.txt")
        if not os.path.exists(file_path):
            print(f"Downloading {split}...")
            r = requests.get(url)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(r.text)


download_ptb()

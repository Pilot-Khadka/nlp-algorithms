import tarfile
import urllib.request
from tqdm import tqdm
from pathlib import Path


class WordNetDownloader:
    URL = "https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz"

    def __init__(self, download_dir="./wordnet_data"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.download_dir / "wordnet-3.0.tar.gz"

    def download(self, force=False):
        if self.filepath.exists() and not force:
            print(f"Using existing: {self.filepath}")
            return self.filepath

        req = urllib.request.Request(self.URL)
        with urllib.request.urlopen(req) as r:
            total = int(r.headers.get("content-length", 0))
            with (
                open(self.filepath, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc="WordNet") as pbar,
            ):
                while chunk := r.read(8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return self.filepath

    def extract(self):
        print("Extracting...")
        with tarfile.open(self.filepath, "r:gz") as tar:
            tar.extractall(self.download_dir)

        dict_dir = next(self.download_dir.rglob("dict"), None)
        return dict_dir or self.download_dir

    def verify(self, dict_dir):
        dict_path = Path(dict_dir)

        if not (dict_path / "index.noun").exists():
            print("Missing WordNet files")
            return False

        exc_files = ["noun.exc", "verb.exc", "adj.exc", "adv.exc"]
        missing = [f for f in exc_files if not (dict_path / f).exists()]

        if missing:
            print(f"Missing exception files: {missing}")
            return False

        print(f"Ready: {dict_path}")
        print(f"Exception lists found: {exc_files}")
        return True

    def get_exceptions(self, dict_dir):
        dict_path = Path(dict_dir)
        exceptions = {}

        for pos in ["noun", "verb", "adj", "adv"]:
            exc_file = dict_path / f"{pos}.exc"
            if exc_file.exists():
                with open(exc_file, "r") as f:
                    exceptions[pos] = [
                        line.strip().split() for line in f if line.strip()
                    ]

        return exceptions

    def install(self, force=False):
        try:
            self.download(force)
            dict_dir = self.extract()
            if self.verify(dict_dir):
                self.exceptions = self.get_exceptions(dict_dir)
                return dict_dir
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download WordNet 3.0")
    parser.add_argument("--dir", default="./wordnet_data", help="Download directory")
    parser.add_argument("--force", action="store_true", help="Force re-download")
    args = parser.parse_args()

    downloader = WordNetDownloader(args.dir)
    result = downloader.install(args.force)

    if result:
        print(f"\nException files location: {result}/")
        print("Files: noun.exc, verb.exc, adj.exc, adv.exc")
    exit(0 if result else 1)

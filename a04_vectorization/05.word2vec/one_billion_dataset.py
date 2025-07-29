from collections import Counter
import os
import tarfile
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_obw(dest_dir="./1billion_word"):
    url = "https://www.statmt.org/lm-benchmark/1-billion-word-language-modeling-benchmark-r13output.tar.gz"
    filename = url.split("/")[-1]
    tar_path = os.path.join(dest_dir, filename)

    if os.path.exists(
        os.path.join(
            dest_dir,
            "1-billion-word-language-modeling-benchmark-r13output",
            "training-monolingual.tokenized.shuffled",
        )
    ):
        print("Dataset already exists. Skipping download.")
        return

    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(tar_path):
        print(f"Downloading dataset from {url} ...")
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=filename
        ) as pbar:
            urllib.request.urlretrieve(
                url, tar_path, reporthook=pbar.update_to)
        print("Download complete.")
    else:
        print("Archive already exists. Skipping download.")

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    os.remove(tar_path)


def load_corpus(data_dir, max_sentences=None):
    token_list = []
    files = sorted(os.listdir(data_dir))
    counter = Counter()
    num_sentences = 0

    for fname in files:
        if not fname.startswith("news.en-"):
            continue

        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
                counter.update(tokens)
                token_list.extend(tokens)
                num_sentences += 1
                if max_sentences and num_sentences >= max_sentences:
                    break
        if max_sentences and num_sentences >= max_sentences:
            break

    return token_list, counter


def build_vocab(counter, min_count=5):
    vocab = {word for word, freq in counter.items() if freq >= min_count}
    word2idx = {word: idx for idx, word in enumerate(sorted(vocab))}
    idx2word = {idx: word for word, idx in word2idx.items()}
    word_freq = [counter[word] for word in sorted(vocab)]
    return word2idx, idx2word, word_freq


def tokens_to_ids(tokens, word2idx):
    return [word2idx[token] for token in tokens if token in word2idx]


if __name__ == "__main__":
    download_and_extract_obw()

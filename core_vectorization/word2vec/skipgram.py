import re
import numpy as np
from collections import deque


class Skipgram:
    def __init__(
        self,
        window_size=2,
        neg_samples=2,
        path="../../data/tiny_shakespeare.txt",
    ):
        self.corpus = []
        self.path = path
        self.vocab = set()
        self.window_size = window_size
        self.neg_samples = neg_samples

        self._read_dataset()
        self._build_vocab()
        self.negative_samples_table = self._create_samples_table()
        self._build_dataset()

    def _read_dataset(self):
        with open(self.path, "r", encoding="utf-8") as file:
            text = file.read()
        text = re.sub(r"\n", " ", text)  # remove newline character
        text = re.sub(r"[^a-zA-Z0-9 ]", "", text)  # remove special characters
        self.corpus = text.split()

    def _build_vocab(self):
        self.word_counts = {}
        for word in self.corpus:
            self.word_counts[word] = self.word_counts.get(word, 0) + 1

        # pyrefly: ignore [bad-assignment]
        self.vocab = sorted(self.word_counts.keys())
        self.word2idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

        # negative sampling strategy
        word_freq = np.array(
            [self.word_counts[word] for word in self.vocab], dtype=np.float32
        )
        word_freq = word_freq ** (3 / 4)
        self.sampling_distribution = word_freq / np.sum(word_freq)

    def _create_samples_table(self, table_size=1e6):
        table_size = int(table_size)
        word_indices = np.arange(len(self.vocab))
        word_probs = (self.sampling_distribution * table_size).astype(int)

        sample_table = np.concatenate(
            [np.full(freq, idx) for idx, freq in zip(word_indices, word_probs)]
        )
        np.random.shuffle(sample_table)
        return sample_table

    def _build_dataset(self, batch_size=64):
        dataset = []
        window = deque(maxlen=2 * self.window_size + 1)
        total_words = len(self.corpus)

        for i in range(total_words):
            current = self.corpus[i]
            window.append(current)

            if len(window) == 2 * self.window_size + 1:
                center_word = window[self.window_size]
                center_idx = self.word2idx[center_word]

                context_words = [
                    window[j] for j in range(len(window)) if j != center_idx
                ]

                for context in context_words:
                    context_idx = self.word2idx[context]
                    dataset.append((center_idx, context_idx, 1))

                for _ in range(self.neg_samples):
                    negative_idx = np.random.choice(
                        self.negative_samples_table, size=1)
                    dataset.append((center_idx, int(negative_idx), 0))
            print(f"\rProcessing {
                  i + 1}/{total_words} words", end="\r", flush=True)

        self.dataset = dataset

    def batch_generator(self, batch_size=64):
        np.random.shuffle(self.dataset)
        for i in range(0, len(self.dataset), batch_size):
            yield self.dataset[i: i + batch_size]

    def __len__(self):
        return len(self.vocab)


def main():
    skipgram = Skipgram()
    # pyrefly: ignore [missing-attribute]
    skipgram.build_dataset()


if __name__ == "__main__":
    main()

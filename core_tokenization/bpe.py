from typing import List, Tuple, Dict
from collections import defaultdict

import json
import heapq
from tqdm import tqdm
from collections import Counter

from engine.registry import register_tokenizer


@register_tokenizer("bpe")
class BPETokenizer:
    def __init__(self, vocab_size: int = 1000, end_of_word: str = "</w>"):
        self.vocab_size = vocab_size
        self.end_of_word = end_of_word
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.merge_ranks: Dict[Tuple[str, str], int] = {}

    def train(self, corpus: str):
        words = corpus.split()
        word_freqs = Counter()

        for word in tqdm(words, desc="Collecting frequencies"):
            chars = list(word) + [self.end_of_word]
            word_freqs[tuple(chars)] += 1

        vocab = set()
        for w in word_freqs:
            vocab.update(w)
        self.vocab = {ch: i for i, ch in enumerate(sorted(vocab))}

        pair_stats = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pair_stats[(word[i], word[i + 1])] += freq

        heap = [(-freq, pair) for pair, freq in pair_stats.items()]
        heapq.heapify(heap)

        pbar = tqdm(total=self.vocab_size - len(self.vocab), desc="Training BPE")

        while len(self.vocab) < self.vocab_size and heap:
            neg_freq, best_pair = heapq.heappop(heap)
            freq = -neg_freq

            if pair_stats.get(best_pair, 0) != freq:
                continue

            self.merges.append(best_pair)
            merged_token = "".join(best_pair)
            self.vocab[merged_token] = len(self.vocab)

            self.merge_ranks[best_pair] = len(self.merge_ranks)

            new_word_freqs = {}
            pair_stats = defaultdict(int)

            for word, count in word_freqs.items():
                new_word = self._merge_word(word, best_pair)
                new_word_freqs[new_word] = count

                for i in range(len(new_word) - 1):
                    pair_stats[(new_word[i], new_word[i + 1])] += count

            word_freqs = new_word_freqs

            heap = [(-f, p) for p, f in pair_stats.items()]
            heapq.heapify(heap)

            pbar.update(1)

        pbar.close()

        if not self.merge_ranks:
            self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def tokenize(self, text: str) -> List[str]:
        if not self.merges:
            raise RuntimeError("Tokenizer must be trained first")

        tokens = []

        for word in text.split():
            word_tuple = list(word) + [self.end_of_word]
            word_tuple = self._apply_merges_greedy(word_tuple)

            tokens.extend(word_tuple)

        return tokens

    def _apply_merges_greedy(self, word: List[str]) -> List[str]:
        word = list(word)

        while True:
            candidates = []
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair in self.merge_ranks:
                    candidates.append((self.merge_ranks[pair], i, pair))

            if not candidates:
                break

            _, idx, pair = min(candidates)

            merged = "".join(pair)
            word = word[:idx] + [merged] + word[idx + 2 :]

        return word

    def _merge_word(
        self, word: Tuple[str, ...], pair: Tuple[str, str]
    ) -> Tuple[str, ...]:
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append("".join(pair))
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        return tuple(new_word)

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        text = text.replace(self.end_of_word, " ")
        return " ".join(text.split())

    def save(self, path: str):
        data = {
            "vocab": self.vocab,
            "merges": self.merges,
            "end_of_word": self.end_of_word,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)

        self.vocab = data["vocab"]

        merges: List[Tuple[str, str]] = []
        for pair in data["merges"]:
            a, b = pair
            merges.append((a, b))

        self.merges = merges
        self.end_of_word = data["end_of_word"]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

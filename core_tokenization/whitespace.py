from typing import List, Dict
from tqdm import tqdm
from collections import Counter

from core_tokenization.base import BaseTokenizer


class WhitespaceTokenizer(BaseTokenizer):
    def __init__(self):
        self.vocab: Dict[str, int]
        self.special_tokens = {"<pad>": 0, "<unk>": 1}

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build_vocab(self, texts: List[str], vocab_size: int = 10000, min_freq: int = 1):
        counter = Counter()
        for text in tqdm(texts, desc="Tokenizing texts for vocab"):
            counter.update(self.tokenize(text))

        most_common = counter.most_common(vocab_size)
        self.vocab = dict(self.special_tokens)
        idx = len(self.vocab)
        for word, freq in tqdm(most_common, desc="Building vocab"):
            if freq >= min_freq:
                self.vocab[word] = idx
                idx += 1

    def encode(self, text: str, max_len: int = 128) -> List[int]:
        tokens = self.tokenize(text)
        unk_id = self.vocab.get("<unk>", 1)
        return [self.vocab.get(t, unk_id) for t in tokens][:max_len]

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseTokenizer(ABC):
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def build_vocab(self, texts: List[str], vocab_size: int = 10000, min_freq: int = 1):
        pass

    @abstractmethod
    def encode(self, text: str, max_len: int = 128) -> List[int]:
        pass

    @abstractmethod
    def get_vocab(self) -> Dict[str, int]:
        pass

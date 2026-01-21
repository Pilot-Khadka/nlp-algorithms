from typing import List, Dict, Optional, Mapping

from collections import Counter


class Vocabulary:
    def __init__(self, token_to_id: Optional[Mapping[str, int]] = None):
        self.token_to_id: Mapping[str, int] = (
            {"<pad>": 0, "<unk>": 1} if token_to_id is None else token_to_id
        )

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.unk_id = self.token_to_id.get("<unk>", 1)
        self.pad_id = self.token_to_id.get("<pad>", 0)

    @classmethod
    def from_tokens(
        cls,
        tokens: List[str],
        vocab_size: int = 10000,
        min_freq: int = 1,
        special_tokens: Optional[Dict[str, int]] = None,
    ):
        if special_tokens is None:
            special_tokens = {"<pad>": 0, "<unk>": 1}

        counter = Counter(tokens)
        token_to_id = dict(special_tokens)
        idx = len(token_to_id)

        for token, freq in counter.most_common(vocab_size):
            if freq >= min_freq and token not in token_to_id:
                token_to_id[token] = idx
                idx += 1

        return cls(token_to_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(id, "<unk>") for id in ids]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

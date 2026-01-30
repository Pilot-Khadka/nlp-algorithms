from typing import List, Optional, Mapping


class Vocabulary:
    def __init__(self, token_to_id: Optional[Mapping[str, int]] = None):
        self.token_to_id: Mapping[str, int] = (
            {"<pad>": 0, "<unk>": 1} if token_to_id is None else token_to_id
        )

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.unk_id = self.token_to_id.get("<unk>", 1)
        self.pad_id = self.token_to_id.get("<pad>", 0)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(id, "<unk>") for id in ids]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id

from typing import List, Optional, Mapping


class Vocabulary:
    def __init__(
        self,
        token_to_id: Optional[Mapping[str, int]] = None,
        special_tokens: Optional[Mapping[str, int]] = None,
    ):
        default_specials = {
            "<pad>": 0,
            "<unk>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }

        if special_tokens is not None:
            default_specials.update(special_tokens)

        if token_to_id is None:
            self.token_to_id = dict(default_specials)
        else:
            self.token_to_id = dict(token_to_id)
            for tok, idx in default_specials.items():
                if tok not in self.token_to_id:
                    self.token_to_id[tok] = len(self.token_to_id)

        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.pad_id = self.token_to_id["<pad>"]
        self.unk_id = self.token_to_id["<unk>"]
        self.sos_id = self.token_to_id["<sos>"]
        self.eos_id = self.token_to_id["<eos>"]

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def decode(self, ids: List[int]) -> List[str]:
        return [self.id_to_token.get(i, "<unk>") for i in ids]

    def __len__(self):
        return len(self.token_to_id)

    def __contains__(self, token):
        return token in self.token_to_id

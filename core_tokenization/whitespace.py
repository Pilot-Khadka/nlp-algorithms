from typing import List


class WhitespaceTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

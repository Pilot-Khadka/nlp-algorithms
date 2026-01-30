from typing import List

from engine.registry import register_tokenizer


@register_tokenizer("whitespace")
class WhitespaceTokenizer:
    @staticmethod
    def tokenize(text: str) -> List[str]:
        return text.lower().split()

    @staticmethod
    def detokenize(tokens: List[str]) -> str:
        return " ".join(tokens)

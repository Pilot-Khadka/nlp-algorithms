from typing import List

from engine.registry import register_tokenizer


@register_tokenizer("whitespace")
class WhitespaceTokenizer:
    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def detokenize(self, tokens: List[str]) -> str:
        return " ".join(tokens)

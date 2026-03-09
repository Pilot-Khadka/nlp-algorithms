"""
This is a naive rule-based implementation of lemmatization.
"""

import re
from .dictionary_lemmas import LEMMAS


def lemmatize_sentence(sentence):
    tokens = re.findall(r"\b\w+\b|[^\w\s]", sentence)
    return [
        lemmatize_word(token) if re.match(r"\w", token) else token for token in tokens
    ]


def lemmatize_word(word):
    original = word
    word_lower = word.lower()

    if word_lower in LEMMAS:
        return LEMMAS[word_lower]

    if len(word_lower) <= 3:
        return original

    lemma = apply_suffix_rules(word_lower)

    return lemma if lemma != word_lower else original


def apply_suffix_rules(word):
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"

    if word.endswith("sses"):
        return word[:-2]

    if word.endswith("ves"):
        if word.endswith("ives"):
            return word[:-3] + "e"
        return word[:-3] + "f"

    if word.endswith("es"):
        if word[-3:-2] in "sxz" or word.endswith(("ches", "shes")):
            return word[:-2]
        if word.endswith("ies"):
            return word[:-3] + "y"
        if len(word) > 3:
            return word[:-1]

    if word.endswith("s") and len(word) > 3:
        if word[-2] not in "aeiou" or word[-2] == "u":
            if not word.endswith("ss"):
                return word[:-1]

    if word.endswith("ingly"):
        stem = word[:-5]
        if len(stem) >= 3:
            return stem + "e" if stem.endswith(("c", "g")) else stem

    if word.endswith("ing"):
        stem = word[:-3]
        if len(stem) >= 3:
            if stem.endswith(stem[-1]) and stem[-1] not in "aeiou" and len(stem) > 1:
                return stem[:-1]
            return stem + "e" if stem.endswith(("c", "g", "v")) else stem

    if word.endswith("edly"):
        stem = word[:-4]
        if len(stem) >= 3:
            return stem + "e" if stem.endswith(("c", "g")) else stem

    if word.endswith("ed"):
        stem = word[:-2]
        if len(stem) >= 3:
            if stem.endswith(stem[-1]) and stem[-1] not in "aeiou" and len(stem) > 1:
                return stem[:-1]
            return stem + "e" if stem.endswith(("c", "g", "v", "t")) else stem

    if word.endswith("ly") and len(word) > 4:
        return word[:-2]

    if word.endswith("er") and len(word) > 4:
        stem = word[:-2]
        if stem.endswith(stem[-1]) and stem[-1] not in "aeiou":
            return stem[:-1]
        return stem

    if word.endswith("est") and len(word) > 5:
        stem = word[:-3]
        if stem.endswith(stem[-1]) and stem[-1] not in "aeiou":
            return stem[:-1]
        return stem

    return word


def main():
    test_sentences = [
        "Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item.",
        "The children were running quickly through the streets.",
        "He has eaten better meals than this, but the worst was yesterday.",
        "The geese flew over the lakes while the mice ran through the houses.",
    ]

    for sentence in test_sentences:
        print(f"Original: {sentence}")
        print(f"Lemmatized: {' '.join(lemmatize_sentence(sentence))}")
        print()


if __name__ == "__main__":
    main()

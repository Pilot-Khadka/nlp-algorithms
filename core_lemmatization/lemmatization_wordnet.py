import os
from collections import defaultdict


class WordNetLemmatizer:
    POS_MAP = {
        "noun": ("noun.exc", "index.noun"),
        "verb": ("verb.exc", "index.verb"),
        "adj": ("adj.exc", "index.adj"),
        "adv": ("adv.exc", "index.adv"),
    }

    NOUN_RULES = [
        ("s", ""),  # cats -> cat
        ("ses", "s"),  # houses -> house
        ("xes", "x"),
        ("zes", "z"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("men", "man"),  # women -> woman
        ("ies", "y"),  # babies -> baby
        ("ves", "f"),  # wolves -> wolf
        ("ves", "fe"),  # knives -> knife
    ]

    VERB_RULES = [
        ("s", ""),  # eats -> eat
        ("ies", "y"),  # tries -> try
        ("es", "e"),  # writes -> write
        ("es", ""),  # does -> do
        ("ed", ""),  # walked -> walk
        ("ed", "e"),  # lived -> live
        ("ing", ""),  # running -> run
        ("ing", "e"),  # making -> make
    ]

    ADJ_RULES = [
        ("er", ""),  # faster -> fast
        ("er", "e"),  # wider -> wide
        ("est", ""),  # fastest -> fast
        ("est", "e"),  # widest -> wide
    ]

    ADJ_EXCEPTIONS = {
        "better": "good",
        "best": "good",
        "worse": "bad",
        "worst": "bad",
    }

    ADV_RULES = [
        ("ly", ""),  # quickly -> quick
    ]

    def __init__(self, wordnet_dir):
        self.wordnet_dir = wordnet_dir
        self.exceptions = defaultdict(dict)
        self.lemmas = defaultdict(set)

        self._load_exceptions()
        self._load_indices()

    def _load_exceptions(self):
        for pos, (exc_file, _) in self.POS_MAP.items():
            path = os.path.join(self.wordnet_dir, exc_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing exception file: {path}")

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        self.exceptions[pos][parts[0]] = parts[1]

    def _load_indices(self):
        for pos, (_, idx_file) in self.POS_MAP.items():
            path = os.path.join(self.wordnet_dir, idx_file)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing index file: {path}")

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith(" "):
                        continue
                    lemma = line.split()[0].replace("_", " ")
                    self.lemmas[pos].add(lemma)

    def _apply_rules(self, word, pos, rules):
        for suffix, replacement in rules:
            if word.endswith(suffix):
                base = word[: -len(suffix)] + replacement
                if base in self.lemmas[pos]:
                    return base
        return None

    def _lemma_adverb(self, word):
        if word in self.exceptions["adv"]:
            return self.exceptions["adv"][word]

        base = self._apply_rules(word, "adj", self.ADV_RULES)
        if base:
            return base

        if word in self.lemmas["adv"]:
            return word

        return word

    def lemmatize(self, word, pos="noun"):
        word = word.lower()

        if pos not in self.POS_MAP:
            pos = "noun"

        if pos == "adj" and word in self.ADJ_EXCEPTIONS:
            return self.ADJ_EXCEPTIONS[word]

        if word in self.exceptions[pos]:
            return self.exceptions[pos][word]

        if word in self.lemmas[pos]:
            return word

        if pos == "noun":
            result = self._apply_rules(word, "noun", self.NOUN_RULES)
        elif pos == "verb":
            result = self._apply_rules(word, "verb", self.VERB_RULES)
        elif pos == "adj":
            result = self._apply_rules(word, "adj", self.ADJ_RULES)
        elif pos == "adv":
            return self._lemma_adverb(word)
        else:
            result = None

        return result if result else word


def demo():
    wordnet_dir = "wordnet_data/dict/"

    if not os.path.exists(wordnet_dir):
        print(f"Error: WordNet directory not found at {wordnet_dir}")
        print("Please update the path to your WordNet installation.")
        return

    lemmatizer = WordNetLemmatizer(wordnet_dir)

    test_cases = [
        ("running", "verb"),
        ("ran", "verb"),
        ("was", "verb"),
        ("geese", "noun"),
        ("mice", "noun"),
        ("children", "noun"),
        ("houses", "noun"),
        ("better", "adj"),
        ("best", "adj"),
        ("quickly", "adv"),
        ("ate", "verb"),
        ("studies", "noun"),
        ("leaves", "noun"),
    ]

    print(f"{'Word':<15} {'POS':<8} {'Lemma':<15}")
    print("-" * 40)

    for word, pos in test_cases:
        result = lemmatizer.lemmatize(word, pos)
        print(f"{word:<15} {pos:<8} {result:<15}")


if __name__ == "__main__":
    demo()

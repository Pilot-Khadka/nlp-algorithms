from collections import Counter


class TextReader:
    def __init__(self, lowercase: bool = True):
        self.lowercase = lowercase

    def read(self, filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    yield self._tokenize(text)

    def _tokenize(self, text: str) -> list[str]:
        if self.lowercase:
            text = text.lower()
        return text.split()


class BagOfWords:
    def __init__(self):
        self.vocabulary: dict[str, int] = {}

    def fit(self, corpus, max_vocab=50000):
        freq = Counter()
        for tokens in corpus:
            freq.update(tokens)

        most_common = freq.most_common(max_vocab)

        self.vocabulary = {tok: i for i, (tok, _) in enumerate(most_common)}
        return self

    def transform(self, corpus):
        for tokens in corpus:
            vector = Counter()
            for token in tokens:
                if token in self.vocabulary:
                    vector[self.vocabulary[token]] += 1
            yield vector

    def fit_transform(self, corpus):
        return self.fit(corpus).transform(corpus)

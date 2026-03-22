from collections import Counter


class NgramGen:
    def __init__(self, n: int) -> None:
        self.n = n

    def generate(self, text: str):
        words = text.split()
        for i in range(len(words) - self.n + 1):
            yield " ".join(words[i : i + self.n])


class NgramBoW:
    def __init__(self, n: int, max_vocab: int = 50000):
        self.n = n
        self.max_vocab = max_vocab
        self.vocab: dict[str, int] = {}
        self.ng = NgramGen(n)

    def fit(self, texts):
        counter = Counter()

        for text in texts:
            for ng in self.ng.generate(text):
                counter[ng] += 1

        most_common = counter.most_common(self.max_vocab)
        self.vocab = {ngram: i for i, (ngram, _) in enumerate(most_common)}
        return self

    def transform(self, texts):
        for text in texts:
            vec = Counter()
            for ngram in self.ng.generate(text):
                if ngram in self.vocab:
                    vec[self.vocab[ngram]] += 1
            yield vec

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)

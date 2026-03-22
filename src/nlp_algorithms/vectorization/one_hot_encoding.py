from collections import Counter


class OneHotEncoder:
    def __init__(self, max_vocab: int = 50000):
        self.max_vocab = max_vocab
        self.vocab: dict[str, int] = {}

    def fit(self, texts):
        freq = Counter()

        for text in texts:
            for word in text.lower().split():
                freq[word] += 1

        most_common = freq.most_common(self.max_vocab)
        self.vocab = {word: idx for idx, (word, _) in enumerate(most_common)}
        return self

    def transform(self, texts):
        vocab_size = len(self.vocab)

        for text in texts:
            words = text.lower().split()
            encoded_sentence = []

            for word in words:
                vector = [0] * vocab_size
                if word in self.vocab:
                    vector[self.vocab[word]] = 1
                encoded_sentence.append(vector)

            yield encoded_sentence

    def transform_sentence(self, texts):
        vocab_size = len(self.vocab)
        for text in texts:
            vec = [0] * vocab_size
            for word in text.lower().split():
                if word in self.vocab:
                    vec[self.vocab[word]] = 1
            yield vec

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)

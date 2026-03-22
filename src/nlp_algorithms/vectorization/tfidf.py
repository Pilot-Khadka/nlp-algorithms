import math
from collections import Counter


class TFIDF:
    def __init__(self):
        self.idf = {}
        self.df = {}
        self.vocab = set()
        self.n_docs = 0

    def fit(self, texts):
        """
        First pass: calc document frequency (df) per word.
        """
        self.n_docs = 0
        df_counter = Counter()

        for text in texts:
            self.n_docs += 1
            words = text.split()
            unique_words = set(words)
            df_counter.update(unique_words)

        self.idf = {
            word: math.log(self.n_docs / (1 + df)) for word, df in df_counter.items()
        }
        self.df = df_counter
        self.vocab = set(df_counter.keys())
        return self

    def transform(self, texts):
        """
        Second pass: calc TF-IDF vectors for each document.
        """
        for text in texts:
            words = text.split()
            if not words:
                yield {}
                continue

            tf_counter = Counter(words)
            total_words = sum(tf_counter.values())

            tfidf_vec = {}
            for word, count in tf_counter.items():
                if word in self.idf:
                    tf = count / total_words
                    tfidf_vec[word] = tf * self.idf[word]

            yield tfidf_vec

    def fit_transform(self, texts):
        texts = list(texts)
        return self.fit(texts).transform(texts)


if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are pets",
    ]

    tfidf = TFIDF()
    vectors = list(tfidf.fit_transform(documents))

    print("Vocabulary:", tfidf.vocab)
    print("IDF:", tfidf.idf)

    for i, vec in enumerate(vectors):
        print(f"\nDocument {i + 1} TF-IDF:")
        print(vec)

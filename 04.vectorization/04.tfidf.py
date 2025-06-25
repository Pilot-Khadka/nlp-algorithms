import math
from collections import defaultdict


class tfidf:
    def __init__(self):
        self.tf = []
        self.idf = {}
        self.vocab = {}

    def fit_transform(self, x):
        N = len(x)
        corpus = []
        vocab = defaultdict(int)

        for doc in x:
            word_counts = defaultdict(int)
            for word in doc.split():
                word_counts[word] += 1
                vocab[word] += 1
            corpus.append(word_counts)

        # calculate document frequency and idf
        for word in vocab:
            df = sum(1 for doc in corpus if word in doc)
            self.idf[word] = math.log(N / (1 + df))

        # tfidf
        tfidf_matrix = []
        for word_counts in corpus:
            doc_tfidf = {}
            total_words = sum(word_counts.values())
            for word, count in word_counts.items():
                tf = count / total_words  # term frequency
                tfidf_value = tf * self.idf[word]  # TF-IDF score
                doc_tfidf[word] = tfidf_value
            tfidf_matrix.append(doc_tfidf)

        return tfidf_matrix


if __name__ == "__main__":
    documents = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "cats and dogs are pets",
    ]

    tfidf_model = tfidf()
    result = tfidf_model.fit_transform(documents)

    print("Input documents:")
    for i, doc in enumerate(documents):
        print(f"  Doc {i + 1}: '{doc}'")

    print(f"\nVocabulary size: {len(tfidf_model.idf)}")
    print("Vocabulary:", list(tfidf_model.idf.keys()))

    print("\nIDF values:")
    for word, idf_val in sorted(tfidf_model.idf.items()):
        print(f"  {word}: {idf_val:.4f}")

    print("\nTF-IDF Results:")
    for i, doc_tfidf in enumerate(result):
        print(f"\nDocument {i + 1}:")
        for word, score in sorted(doc_tfidf.items(), key=lambda x: x[1], reverse=True):
            print(f"  {word}: {score:.4f}")

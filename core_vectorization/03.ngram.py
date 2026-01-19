def generate_ngrams(text, n):
    words = text.split()
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = " ".join(words[i : i + n])
        ngrams.append(ngram)
    return ngrams


def create_vocab(texts, n):
    all_ngrams = set()
    for text in texts:
        ngrams = generate_ngrams(text, n)
        all_ngrams.update(ngrams)

    vocab = {ngram: idx for idx, ngram in enumerate(sorted(all_ngrams))}
    return vocab


def bow(text, all_encodings, n=2):
    all_texts = []
    ngrams = generate_ngrams(text, n)

    for ngram in ngrams:
        encoded_text = [0] * len(all_encodings)
        print(f"{n}-gram:", ngram)

        if ngram not in all_encodings.keys():
            print("ngram not in dictionary")
            continue
        else:
            print(f"index: {all_encodings[ngram]}")
            encoded_text[all_encodings[ngram]] = 1
            all_texts.append(encoded_text)

    return all_texts


def bow_sentence(text, all_encodings, n=2):
    encoded_text = [0] * len(all_encodings)
    ngrams = generate_ngrams(text, n)

    for ngram in ngrams:
        if ngram in all_encodings:
            encoded_text[all_encodings[ngram]] += 1  # Count occurrences

    return encoded_text


if __name__ == "__main__":
    d1 = "hello there how are you?"
    d2 = "I hope you are doing well."
    d3 = "There is more sand grains the deserts of the earth than there are stars in the universe"

    all_texts = [d1, d2, d3]
    combined_text = d1 + " " + d2 + " " + d3

    print("=" * 50)
    print("UNIGRAMS (n=1)")
    print("=" * 50)

    unigram_vocab = create_vocab(all_texts, n=1)
    print(f"Unigram vocabulary size: {len(unigram_vocab)}")
    print("Vocabulary:", list(unigram_vocab.keys())[:10], "...")  # Show first 10

    print(f"\nProcessing: '{d1}'")
    unigram_result = bow(d1, unigram_vocab, n=1)
    print(f"Number of unigram vectors: {len(unigram_result)}")

    sentence_unigram = bow_sentence(d1, unigram_vocab, n=1)
    print(f"Sentence vector (first 10 dims): {sentence_unigram[:10]}")

    print("\n" + "=" * 50)
    print("BIGRAMS (n=2)")
    print("=" * 50)

    bigram_vocab = create_vocab(all_texts, n=2)
    print(f"Bigram vocabulary size: {len(bigram_vocab)}")
    print("Vocabulary:", list(bigram_vocab.keys())[:10], "...")  # Show first 10

    print(f"\nProcessing: '{d1}'")
    bigram_result = bow(d1, bigram_vocab, n=2)
    print(f"Number of bigram vectors: {len(bigram_result)}")

    sentence_bigram = bow_sentence(d1, bigram_vocab, n=2)
    print(f"Sentence vector (first 10 dims): {sentence_bigram[:10]}")

    print("\n" + "=" * 50)
    print("TRIGRAMS (n=3)")
    print("=" * 50)

    trigram_vocab = create_vocab(all_texts, n=3)
    print(f"Trigram vocabulary size: {len(trigram_vocab)}")
    print("Vocabulary:", list(trigram_vocab.keys())[:10], "...")  # Show first 10

    print(f"\nProcessing: '{d1}'")
    trigram_result = bow(d1, trigram_vocab, n=3)
    print(f"Number of trigram vectors: {len(trigram_result)}")

    # Sentence-level representation
    sentence_trigram = bow_sentence(d1, trigram_vocab, n=3)
    print(f"Sentence vector (first 10 dims): {sentence_trigram[:10]}")

    print("\n" + "=" * 50)
    print("EXAMPLE: Processing all texts with bigrams")
    print("=" * 50)

    for i, text in enumerate(all_texts, 1):
        print(f"\nText {i}: '{text}'")
        result = bow_sentence(text, bigram_vocab, n=2)
        non_zero_indices = [j for j, val in enumerate(result) if val > 0]
        print(f"Non-zero dimensions: {len(non_zero_indices)}")
        if non_zero_indices:
            print(
                "Active bigrams:",
                [list(bigram_vocab.keys())[j] for j in non_zero_indices[:5]],
                "...",
            )

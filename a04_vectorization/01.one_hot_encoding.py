def create_vocabulary(texts):
    if isinstance(texts, str):
        texts = [texts]

    vocab = set()
    for text in texts:
        words = text.lower().split()
        vocab.update(words)

    word_to_idx = {}
    for idx, word in enumerate(sorted(vocab)):
        word_to_idx[word] = idx

    return word_to_idx


def one_hot_encode(text, word_to_idx):
    encoded_text = []
    words = text.lower().split()
    vocab_size = len(word_to_idx)

    for word in words:
        one_hot = [0] * vocab_size
        if word in word_to_idx:
            one_hot[word_to_idx[word]] = 1
        else:
            print(f"{word} not found in vocab")
        encoded_text.append(one_hot)

    return encoded_text


def one_hot_encode_sentence(text, word_to_idx):
    vocab_size = len(word_to_idx)
    one_hot = [0] * vocab_size
    words = text.lower().split()

    for word in words:
        if word in word_to_idx:
            one_hot[word_to_idx[word]] = 1  # Set to 1 if word is present

    return one_hot


def display_encoding(text, encoded_vectors, word_to_idx):
    words = text.lower().split()

    print(f"Text: '{text}'")
    print(f"Words: {words}")
    print("Encoded vectors:")

    for i, (word, vector) in enumerate(zip(words, encoded_vectors)):
        active_idx = vector.index(1) if 1 in vector else -1
        print(
            f"  Word '{word}' -> Index {active_idx} -> {vector[:10]}{
                '...' if len(vector) > 10 else ''
            }"
        )


if __name__ == "__main__":
    d1 = "hello there how are you?"
    d2 = "I hope you are doing well."
    d3 = "There is more sand grains the deserts of the earth than there are stars in the universe"

    all_texts = [d1, d2, d3]
    combined_text = d1 + " " + d2 + " " + d3

    print("=" * 60)
    print("ORIGINAL IMPLEMENTATION FIXED")
    print("=" * 60)

    word_to_idx = create_vocabulary(all_texts)
    vocab = set(word.lower() for text in all_texts for word in text.split())

    print(f"Combined text: '{combined_text}'")
    print(f"Length of original text (words): {len(combined_text.split())}")
    print(f"Length of unique words: {len(vocab)}")
    print(f"Vocabulary (first 10): {sorted(list(vocab))[:10]}")

    print("\n" + "=" * 60)
    print("ONE-HOT ENCODING EXAMPLES")
    print("=" * 60)

    for i, text in enumerate(all_texts, 1):
        print(f"\nDocument {i}:")
        encoded = one_hot_encode(text, word_to_idx)
        display_encoding(text, encoded, word_to_idx)

        sentence_encoding = one_hot_encode_sentence(text, word_to_idx)
        active_words = [
            word for word, idx in word_to_idx.items() if sentence_encoding[idx] == 1
        ]
        print(f"  Sentence encoding: {len(active_words)} unique words active")
        print(
            f"  Active words: {sorted(active_words)[:10]}{
                '...' if len(active_words) > 10 else ''
            }"
        )

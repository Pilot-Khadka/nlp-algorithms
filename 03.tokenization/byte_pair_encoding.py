class BPE:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}

    def train(self, text):
        tokens = list(text.encode("utf-8"))
        self.vocab = {i: bytes([i]) for i in range(256)}
        current_vocab_size = 256
        while current_vocab_size < self.vocab_size:
            pair_counts = get_common_pair(tokens)

            if not pair_counts:
                break

            most_common_pair = max(pair_counts, key=pair_counts.get)
            new_token_idx = current_vocab_size
            self.merges[most_common_pair] = new_token_idx
            self.vocab[new_token_idx] = (
                self.vocab[most_common_pair[0]] +
                self.vocab[most_common_pair[1]]
            )
            tokens = merge(tokens, most_common_pair, new_token_idx)
            current_vocab_size += 1

    def encode(self, text):
        tokens = list(text.encode("utf-8"))

        while True:
            pair_counts = get_common_pair(tokens)
            possible_merges = []
            for pair in pair_counts:
                if pair in self.merges:
                    possible_merges.append((pair, self.merges[pair]))

            if not possible_merges:
                break

            pair_to_merge = min(possible_merges, key=lambda x: x[1])[0]
            merge_idx = self.merges[pair_to_merge]
            tokens = merge(tokens, pair_to_merge, merge_idx)

        return tokens

    def decode(self, tokens):
        byte_data = b"".join(self.vocab[token] for token in tokens)
        return byte_data.decode("utf-8", errors="replace")


def get_common_pair(tokens):
    count = {}
    for i in range(len(tokens) - 1):
        pair = (tokens[i], tokens[i + 1])
        count[pair] = count.get(pair, 0) + 1
    return count


def merge(ids, pair, idx):
    new_list = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_list.append(idx)
            i += 2
        else:
            new_list.append(ids[i])
            i += 1
    return new_list


if __name__ == "__main__":
    bpe = BPE(vocab_size=300)

    text = "This was not the map we found in Billy Bones’s chest, but an accurate copy, complete in all things names and heights and soundings with the single exception of the red crosses and the written notes."
    bpe.train(text)

    test_text = "hello world"
    encoded = bpe.encode(test_text)
    print(f"Original: {test_text}")
    print(f"Encoded: {encoded}")

    decoded = bpe.decode(encoded)
    print(f"Decoded: {decoded}")

    print(f"\nLearned {len(bpe.merges)} merges:")
    for i, (pair, idx) in enumerate(list(bpe.merges.items())[:5]):
        pair_bytes = bpe.vocab[pair[0]] + bpe.vocab[pair[1]]
        print(
            f"  {pair} -> {idx} ('{pair_bytes.decode('utf-8', errors='replace')}')")

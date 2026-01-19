from typing import List, Dict
from tqdm import tqdm


from core_tokenization.base import BaseTokenizer


class BPE(BaseTokenizer):
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {}
        self.str_to_token = {}

    def tokenize(self, text: str) -> List[str]:
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

        return [self.vocab[token].decode("utf-8", errors="replace") for token in tokens]

    def build_vocab(self, texts: List[str], vocab_size: int = 10000, min_freq: int = 1):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        if isinstance(texts, str):
            texts = [texts]

        all_tokens = []
        for text in texts:
            all_tokens.extend(list(text.encode("utf-8")))

        current_vocab_size = 256
        num_merges = self.vocab_size - 256

        pbar = tqdm(total=num_merges, desc="Building BPE vocabulary")

        while current_vocab_size < self.vocab_size:
            pair_counts = get_common_pair(all_tokens)

            if not pair_counts:
                break

            filtered_pairs = {
                pair: count for pair, count in pair_counts.items() if count >= min_freq
            }

            if not filtered_pairs:
                break

            most_common_pair = max(filtered_pairs.items(), key=lambda item: item[1])[0]
            new_token_idx = current_vocab_size

            self.merges[most_common_pair] = new_token_idx
            self.vocab[new_token_idx] = (
                self.vocab[most_common_pair[0]] + self.vocab[most_common_pair[1]]
            )

            all_tokens = merge(all_tokens, most_common_pair, new_token_idx)
            current_vocab_size += 1
            pbar.update(1)

        pbar.close()
        self._build_str_to_token_map()

    def _build_str_to_token_map(self):
        self.str_to_token = {}
        for token_idx, byte_seq in self.vocab.items():
            token_str = byte_seq.decode("utf-8", errors="replace")
            self.str_to_token[token_str] = token_idx

    def encode(self, text: str, max_len: int = 128) -> List[int]:
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

        return tokens[:max_len]

    def decode(self, tokens: List[int]) -> str:
        byte_data = b"".join(self.vocab[token] for token in tokens)
        return byte_data.decode("utf-8", errors="replace")

    def get_vocab(self) -> Dict[str, int]:
        if not self.str_to_token:
            self._build_str_to_token_map()
        return self.str_to_token.copy()


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

    text = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things names and heights and soundings with the single exception of the red crosses and the written notes."

    bpe.build_vocab([text], vocab_size=300)

    test_text = "hello world"
    encoded = bpe.encode(test_text)
    print(f"\nOriginal: {test_text}")
    print(f"Encoded: {encoded}")
    decoded = bpe.decode(encoded)
    print(f"Decoded: {decoded}")

    print(f"\nVocabulary size: {len(bpe.vocab)}")
    print(f"Learned {len(bpe.merges)} merges")
    print("\nFirst 10 merges:")
    for i, (pair, idx) in enumerate(list(bpe.merges.items())[:10]):
        pair_bytes = bpe.vocab[pair[0]] + bpe.vocab[pair[1]]
        print(f"  {pair} -> {idx} ('{pair_bytes.decode('utf-8', errors='replace')}')")

    vocab_dict = bpe.get_vocab()
    print("\nSample vocabulary entries:")
    for i, (token_str, token_id) in enumerate(list(vocab_dict.items())[:10]):
        print(f"  '{token_str}' -> {token_id}")

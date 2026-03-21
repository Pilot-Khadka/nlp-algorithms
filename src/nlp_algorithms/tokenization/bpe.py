import json
import heapq
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from nlp_algorithms.engine.registry import register_tokenizer


@register_tokenizer("bpe")
class BytePairEncoder:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256
        self.merges = {}  # (token_id1, token_id2) -> new_token_id
        self.vocab = {i: bytes([i]) for i in range(256)}  # id -> bytes

    def train(self, text: str, verbose: bool = False):
        ids = list(text.encode("utf-8"))
        n = len(ids)
        print(f"Initial sequence length: {n:,} tokens")

        tokens = list(ids)
        prev = list(range(-1, n - 1))
        nxt = list(range(1, n)) + [n]
        pair_counts = defaultdict(int)
        pair_positions = defaultdict(set)
        for i in range(n - 1):
            pair = (tokens[i], tokens[nxt[i]])
            pair_counts[pair] += 1
            pair_positions[pair].add(i)

        heap = [(-count, pair) for pair, count in pair_counts.items()]
        heapq.heapify(heap)

        seq_len = n
        # outer_bar = tqdm(range(self.num_merges), disable=not verbose, desc="BPE merges")
        for merge_idx in range(self.num_merges):
            while heap:
                neg_count, best_pair = heapq.heappop(heap)
                if pair_counts.get(best_pair, 0) == -neg_count:
                    break
            else:
                break

            best_count = -neg_count
            if best_count < 2:
                print("Stopping early: no pairs with frequency >= 2")
                break

            new_id = 256 + merge_idx
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            positions = sorted(pair_positions.pop(best_pair, set()))
            pair_counts[best_pair] = 0

            for i in positions:
                if tokens[i] != best_pair[0]:
                    continue
                j = nxt[i]
                if j == n or tokens[j] != best_pair[1]:
                    continue

                p = prev[i]
                if p != -1:
                    self._update_pair(
                        pair_counts=pair_counts,
                        pair_positions=pair_positions,
                        heap=heap,
                        old=(tokens[p], tokens[i]),
                        old_pos=p,
                        new=(tokens[p], new_id),
                        new_pos=p,
                    )

                k = nxt[j]
                if k != n:
                    self._update_pair(
                        pair_counts=pair_counts,
                        pair_positions=pair_positions,
                        heap=heap,
                        old=(tokens[j], tokens[k]),
                        old_pos=j,
                        new=(new_id, tokens[k]),
                        new_pos=i,
                    )

                tokens[i] = new_id
                tokens[j] = None
                nxt[i] = k
                if k != n:
                    prev[k] = i
                seq_len -= 1

            if verbose:
                print(
                    f"Merge {merge_idx + 1}: {best_pair} -> {new_id} "
                    f"(freq={best_count}, new_len={seq_len})"
                )

        print(f"Final sequence length: {seq_len} tokens")
        print(f"Compression ratio: {len(text.encode('utf-8')) / seq_len:.2f}x")
        return self

    def _update_pair(
        self,
        pair_counts,
        pair_positions,
        heap,
        old,
        old_pos,
        new,
        new_pos,
    ):
        """Remove old pair at old_pos, add new pair at new_pos."""
        if old_pos in pair_positions[old]:
            pair_counts[old] -= 1
            pair_positions[old].remove(old_pos)
            heapq.heappush(heap, (-pair_counts[old], old))

        pair_counts[new] += 1
        pair_positions[new].add(new_pos)
        heapq.heappush(heap, (-pair_counts[new], new))

    def tokenize(self, text: str):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            best_pair = min(
                zip(ids, ids[1:]),
                key=lambda p: self.merges.get(p, float("inf")),
            )
            if best_pair not in self.merges:
                break
            ids = self._merge_list(ids, best_pair, self.merges[best_pair])
        return ids

    @staticmethod
    def _merge_list(ids, pair, new_id):
        """Replace all non-overlapping occurrences of pair with new_id (left-to-right)."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def detokenize(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path: str | Path):
        data = {"vocab": self.vocab, "merges": self.merges}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str | Path):
        with open(path, "r") as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [(int(p[0]), int(p[1])) for p in data["merges"]]


def pretty_vocab(vocab):
    def fmt_byte(b):
        # printable ASCII range
        if 32 <= b <= 126:
            return chr(b)
        return f"<{b:02x}>"

    def fmt_bytes(bs):
        return "".join(fmt_byte(b) for b in bs)

    base = []
    merged = []

    for tok_id, seq in vocab.items():
        (base if len(seq) == 1 else merged).append((tok_id, seq))

    print("\n--- Base tokens (single bytes) ---")
    for tok_id, seq in sorted(base):
        readable = fmt_bytes(seq)
        print(f"{tok_id:4d}: {seq!r:12s} -> {readable}")

    print("\n--- Merged tokens ---")
    for tok_id, seq in sorted(merged):
        readable = fmt_bytes(seq)
        print(f"{tok_id:4d}: {seq!r:12s} -> {readable}")


if __name__ == "__main__":
    import time
    from nlp_algorithms.util.path_util import get_data_path

    sample_txt_file = get_data_path() / "shakespeare.txt"
    text = open(sample_txt_file, "r", encoding="utf-8").read()
    t0 = time.time()
    tokenizer = BytePairEncoder(vocab_size=512)
    tokenizer.train(text, verbose=False)
    t1 = time.time()
    print("Tokenizer vocab:")
    pretty_vocab(tokenizer.vocab)

    test_strings = [
        "hello world",
        "To be, or not to be.",
        "The quick brown fox jumps over the lazy dog!",
    ]

    for s in test_strings:
        print(f"\nInput: {s!r}")
        tok = tokenizer.tokenize(s)
        print("Tokenized IDs:", tok)
        detok = tokenizer.detokenize(tok)
        print("Detokenized:", detok)
        if detok == s:
            print("ok")
        else:
            print("mismatch")

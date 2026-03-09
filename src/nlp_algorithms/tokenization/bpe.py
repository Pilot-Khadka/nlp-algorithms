from typing import List, Tuple, Dict
from collections import defaultdict, Counter
import json
from tqdm import tqdm
import heapq

from nlp_algorithms.engine.registry import register_tokenizer


@register_tokenizer("bpe")
class BPETokenizer:
    def __init__(self, vocab_size: int = 10000, end_token: str = "</w>"):
        self.vocab_size = vocab_size
        self.end_token = end_token
        self.id2sym: List[str] = []
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[int, int]] = []
        self.merge_ranks: Dict[Tuple[int, int], int] = {}

    def _get_id(self, sym: str) -> int:
        if sym not in self.vocab:
            idx = len(self.id2sym)
            self.id2sym.append(sym)
            self.vocab[sym] = idx
        return self.vocab[sym]

    def train(self, corpus: str):
        word_freq = Counter(corpus.split())

        words = []
        freqs = []
        for w, cnt in tqdm(word_freq.items(), desc="Building words"):
            ids = [self._get_id(c) for c in w] + [self._get_id(self.end_token)]
            words.append(ids)
            freqs.append(cnt)

        pair_counts = Counter()  # (a,b) -> total frequency
        pair_locs = defaultdict(list)  # (a,b) -> [(word_idx, pos), ...]

        for wi, word in tqdm(enumerate(words), desc="Counting pairs", total=len(words)):
            if len(word) < 2:
                continue
            wf = freqs[wi]
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_counts[pair] += wf
                pair_locs[pair].append((wi, i))

        heap = [(-cnt, pair) for pair, cnt in pair_counts.items()]
        heapq.heapify(heap)

        vocab_size_target = self.vocab_size
        id2sym = self.id2sym

        pbar = tqdm(total=vocab_size_target - len(id2sym), desc="Merging BPEs")
        while len(id2sym) < vocab_size_target and heap:
            neg_cnt, pair = heapq.heappop(heap)
            cnt = -neg_cnt

            if pair_counts.get(pair) != cnt:
                continue
            if cnt <= 0:
                continue

            a, b = pair
            self.merges.append(pair)
            merged_sym = id2sym[a] + id2sym[b]
            m = self._get_id(merged_sym)
            self.merge_ranks[pair] = len(self.merge_ranks)

            # group locations by word to process in reverse index order
            # (reverse to avoid index shifting issues when deleting)
            word_pos = defaultdict(list)
            for wi, pos in pair_locs[pair]:
                word_pos[wi].append(pos)
            pair_locs[pair] = []  # Free memory

            for wi, positions in word_pos.items():
                word = words[wi]
                wf = freqs[wi]
                for pos in sorted(positions, reverse=True):
                    if pos >= len(word) - 1 or word[pos] != a or word[pos + 1] != b:
                        continue

                    if pos > 0:
                        left_pair = (word[pos - 1], a)
                        pair_counts[left_pair] -= wf
                        heapq.heappush(heap, (-pair_counts[left_pair], left_pair))

                    if pos + 2 < len(word):
                        right_pair = (b, word[pos + 2])
                        pair_counts[right_pair] -= wf
                        heapq.heappush(heap, (-pair_counts[right_pair], right_pair))

                    word[pos] = m
                    del word[pos + 1]

                    if pos > 0:
                        new_left = (word[pos - 1], m)
                        pair_counts[new_left] += wf
                        pair_locs[new_left].append((wi, pos - 1))
                        heapq.heappush(heap, (-pair_counts[new_left], new_left))

                    if pos + 1 < len(word):
                        new_right = (m, word[pos + 1])
                        pair_counts[new_right] += wf
                        pair_locs[new_right].append((wi, pos))
                        heapq.heappush(heap, (-pair_counts[new_right], new_right))

            del pair_counts[pair]

            pbar.update(1)
        pbar.close()

        if not self.merge_ranks:
            self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    def _apply_merges(self, ids: List[int]) -> List[int]:
        if not self.merges or len(ids) < 2:
            return ids

        tokens = ids[:]
        n = len(tokens)

        prev = list(range(-1, n - 1))
        nxt = list(range(1, n + 1))
        prev[0] = -1
        nxt[-1] = -1
        valid = [True] * n

        heap = []
        mget = self.merge_ranks.get
        for i in range(n - 1):
            rank = mget((tokens[i], tokens[i + 1]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, pos = heapq.heappop(heap)
            if not valid[pos]:
                continue
            j = nxt[pos]
            if j == -1 or not valid[j]:
                continue
            if mget((tokens[pos], tokens[j])) != rank:
                continue

            a, b = tokens[pos], tokens[j]
            new_sym = self.id2sym[a] + self.id2sym[b]
            if new_sym in self.vocab:
                new_id = self.vocab[new_sym]
            else:
                new_id = len(self.id2sym)
                self.id2sym.append(new_sym)
                self.vocab[new_sym] = new_id

            tokens[pos] = new_id
            valid[j] = False

            new_nxt = nxt[j]
            nxt[pos] = new_nxt
            if new_nxt != -1:
                prev[new_nxt] = pos

            left = prev[pos]
            if left != -1 and valid[left]:
                new_rank = mget((tokens[left], new_id))
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, left))

            if new_nxt != -1:
                new_rank = mget((new_id, tokens[new_nxt]))
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, pos))

        result = []
        idx = 0
        while idx != -1:
            result.append(tokens[idx])
            idx = nxt[idx]
        return result

    def tokenize(self, text: str) -> List[str]:
        if not self.merges:
            raise RuntimeError("Train first before tokenizing.")
        output = []
        for word in text.split():
            ids = []
            for c in word:
                if c in self.vocab:
                    ids.append(self.vocab[c])

            ids.append(self.vocab[self.end_token])
            merged_ids = self._apply_merges(ids)
            output.extend(self.id2sym[i] for i in merged_ids)
        return output

    def detokenize(self, tokens: List[str]) -> str:
        text = "".join(tokens)
        return text.replace(self.end_token, " ").strip()

    def save(self, path: str):
        data = {
            "id2sym": self.id2sym,
            "vocab": self.vocab,
            "merges": self.merges,
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.id2sym = data["id2sym"]
        self.vocab = data["vocab"]
        self.merges = [(int(p[0]), int(p[1])) for p in data["merges"]]
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

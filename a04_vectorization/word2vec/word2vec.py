import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


class SkipGramDataset(Dataset):
    def __init__(self, token_ids, window_size=5):
        self.pairs = []

        for idx in tqdm(range(len(token_ids)), desc="Generating pairs"):
            center = token_ids[idx]
            start = max(0, idx - window_size)
            end = min(len(token_ids), idx + window_size + 1)

            for context_idx in range(start, end):
                if context_idx != idx:  # skip the center word itself
                    context = token_ids[context_idx]
                    self.pairs.append((center, context))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


class UnigramSampler:
    def __init__(self, word_freq, power=0.75):
        self.vocab_size = len(word_freq)
        freqs = np.array(word_freq)
        self.prob_dist = freqs**power
        self.prob_dist /= self.prob_dist.sum()

        # alias table for faster sampling
        self._setup_alias_table()

    def _setup_alias_table(self):
        """alias table for O(1) sampling - Walker's alias method"""
        n = len(self.prob_dist)
        self.alias = np.zeros(n, dtype=np.int32)
        self.prob = np.zeros(n, dtype=np.float32)

        prob_scaled = self.prob_dist * n

        small = []
        large = []

        for i, p in enumerate(prob_scaled):
            if p < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx = small.pop()
            large_idx = large.pop()

            self.prob[small_idx] = prob_scaled[small_idx]
            self.alias[small_idx] = large_idx

            prob_scaled[large_idx] = (
                prob_scaled[large_idx] + prob_scaled[small_idx] - 1.0
            )

            if prob_scaled[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            self.prob[large_idx] = 1.0

        while small:
            small_idx = small.pop()
            self.prob[small_idx] = 1.0

    def sample(self, batch_size, num_negatives):
        total_samples = batch_size * num_negatives

        indices = np.random.randint(0, self.vocab_size, size=total_samples)
        rand_probs = np.random.rand(total_samples)

        mask = rand_probs < self.prob[indices]
        samples = np.where(mask, indices, self.alias[indices])

        return torch.LongTensor(samples.reshape(batch_size, num_negatives))


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        self._init_emb()

    def _init_emb(self):
        bound = 0.5 / self.embedding_dim
        nn.init.uniform_(self.input_embeddings.weight, -bound, bound)
        nn.init.uniform_(self.output_embeddings.weight, -bound, bound)

    def forward(self, center, context, negatives):
        """
        center: (B,)DataLoader
        context: (B,)
        negatives: (B, K)
        """
        center_emb = self.input_embeddings(center)  # (B, D)
        pos_emb = self.output_embeddings(context)  # (B, D)
        neg_emb = self.output_embeddings(negatives)  # (B, K, D)

        # positive score: dot product between center and context
        pos_score = torch.sum(center_emb * pos_emb, dim=1)  # (B,)
        pos_loss = F.logsigmoid(pos_score)

        # negative score: dot product between center and negative samples
        center_emb_expanded = center_emb.unsqueeze(1)  # (B, 1, D)
        # bmm: (B, K, D) x (B, D, 1) -> (B, K, 1)
        neg_score = torch.bmm(neg_emb, center_emb_expanded.transpose(1, 2)).squeeze(
            2
        )  # (B, K)
        neg_loss = torch.sum(F.logsigmoid(-neg_score), dim=1)  # (B,)

        return -torch.mean(pos_loss + neg_loss)

    def get_input_embeddings(self):
        return self.input_embeddings.weight.detach()

    def get_output_embeddings(self):
        return self.output_embeddings.weight.detach()

    def save_model(self, filepath):
        checkpoint = {
            "model_class": "word2vec",
            "model_args": {
                "vocab_size": self.vocab_size,
                "embedding_dim": self.embedding_dim,
            },
            "model_state_dict": self.state_dict(),
        }
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device="cpu"):
        checkpoint = torch.load(filepath, map_location=device)
        model = cls(checkpoint["vocab_size"], checkpoint["embedding_dim"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        print(f"Model loaded from {filepath}")
        return model

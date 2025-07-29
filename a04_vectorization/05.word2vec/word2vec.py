import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from one_billion_dataset import (
    load_corpus,
    build_vocab,
    tokens_to_ids,
    download_and_extract_obw,
)


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
        center: (B,)
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
        # Reshape center_emb for bmm: (B, 1, D)
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
            "model_state_dict": self.state_dict(),
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
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


def save_vocab(word2idx, idx2word, word_freq, filepath):
    vocab_data = {"word2idx": word2idx,
                  "idx2word": idx2word, "word_freq": word_freq}
    with open(filepath, "wb") as f:
        pickle.dump(vocab_data, f)
    print(f"Vocabulary saved to {filepath}")


def load_vocab(filepath):
    with open(filepath, "rb") as f:
        vocab_data = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab_data["word2idx"], vocab_data["idx2word"], vocab_data["word_freq"]


def train_word2vec(
    token_ids,
    word_freq,
    vocab_size,
    embedding_dim=100,
    window_size=5,
    num_negatives=5,
    batch_size=512,
    epochs=5,
    device="cuda",
    save_dir="./models",
    save_every=1,
):
    dataset = SkipGramDataset(token_ids, window_size)
    num_workers = min(4, os.cpu_count()) if os.cpu_count() else 0
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device == "cuda" else False,
        drop_last=True,  # drop last incomplete batch for consistent negative sampling
    )

    model = Word2Vec(vocab_size, embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=2, gamma=0.8)

    sampler = UnigramSampler(word_freq)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (centers, contexts) in enumerate(pbar):
            centers, contexts = (
                centers.to(device, non_blocking=True),
                contexts.to(device, non_blocking=True),
            )

            # sample negatives on CPU to avoid GPU memory issues
            negatives = sampler.sample(len(centers), num_negatives).to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            loss = model(centers, contexts, negatives)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()

            pbar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Avg Loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "LR": f"{optimizer.param_groups[0]['lr']:.6f}",
                }
            )

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(
            f"Epoch {epoch + 1}/{epochs} completed - Average Loss: {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                save_dir, f"word2vec_epoch_{epoch + 1}.pt")
            model.save_model(checkpoint_path)

    final_model_path = os.path.join(save_dir, "word2vec_final.pt")
    model.save_model(final_model_path)

    return model


def find_similar_words(model, word, word2idx, idx2word, top_k=10, device="cpu"):
    if word not in word2idx:
        print(f"Word '{word}' not in vocabulary")
        return []

    model.eval()
    word_idx = word2idx[word]
    embeddings = model.get_input_embeddings().to(device)

    target_emb = embeddings[word_idx].unsqueeze(0)  # (1, D)
    similarities = F.cosine_similarity(target_emb, embeddings, dim=1)
    top_indices = similarities.topk(top_k + 1)[1][1:]  # Skip the word itself

    similar_words = []
    for idx in top_indices:
        similar_word = idx2word[idx.item()]
        similarity_score = similarities[idx].item()
        similar_words.append((similar_word, similarity_score))

    return similar_words


def main():
    data_dir = "./1billion_word/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled"
    save_dir = "./models"
    os.makedirs(save_dir, exist_ok=True)

    tokens, counter = load_corpus(data_dir, max_sentences=100000)
    word2idx, idx2word, word_freq = build_vocab(counter, min_count=5)
    token_ids = tokens_to_ids(tokens, word2idx)

    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Total tokens: {len(token_ids):,}")

    vocab_path = os.path.join(save_dir, "vocab.pkl")
    save_vocab(word2idx, idx2word, word_freq, vocab_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device:{device}")
    model = train_word2vec(
        token_ids=token_ids,
        word_freq=word_freq,
        vocab_size=len(word2idx),
        embedding_dim=100,
        window_size=5,
        num_negatives=5,
        batch_size=1024,
        epochs=5,
        device=device,
        save_dir=save_dir,
        save_every=1,
    )

    print("\nTesting model with similarity queries:")
    test_words = ["the", "king", "good", "run"]
    for word in test_words:
        if word in word2idx:
            similar = find_similar_words(
                model, word, word2idx, idx2word, top_k=5, device=device
            )
            print(f"\nWords similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")


if __name__ == "__main__":
    download_and_extract_obw()
    main()

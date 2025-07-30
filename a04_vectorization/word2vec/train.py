import os
import pickle
import torch
import argparse

from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from one_billion_dataset import (
    load_corpus,
    build_vocab,
    tokens_to_ids,
    download_and_extract_obw,
)
from word2vec import UnigramSampler, Word2Vec, SkipGramDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec on 1Billion Word Dataset"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./1billion_word/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled",
        help="Directory containing tokenized and shuffled training data",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../../models/word2vec/",
        help="Directory to save model and vocab",
    )
    parser.add_argument(
        "--max_sentences",
        type=int,
        default=100000,
        help="Limit number of sentences to load",
    )
    parser.add_argument(
        "--min_count",
        type=int,
        default=5,
        help="Minimum word frequency to include in vocab",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=100, help="Embedding dimension size"
    )
    parser.add_argument(
        "--window_size", type=int, default=5, help="Context window size for skip-gram"
    )
    parser.add_argument(
        "--num_negatives", type=int, default=5, help="Number of negative samples"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1024, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cuda or cpu",
    )
    return parser.parse_args()


def add_special_tokens_to_vocab(
    word2idx,
    idx2word,
    word_freq,
    special_tokens=["<pad>", "<unk>", "<eos>"],
):
    new_word2idx = {}
    new_idx2word = {}
    new_word_freq = {}

    offset = len(special_tokens)

    for i, token in enumerate(special_tokens):
        new_word2idx[token] = i
        new_idx2word[i] = token
        new_word_freq[token] = 0

    for word, old_idx in word2idx.items():
        new_idx = old_idx + offset
        new_word2idx[word] = new_idx
        new_idx2word[new_idx] = word
        new_word_freq[word] = word_freq[word]

    return new_word2idx, new_idx2word, new_word_freq


def save_vocab(word2idx, idx2word, word_freq, filepath):
    word2idx, idx2word, word_freq = add_special_tokens_to_vocab(
        word2idx, idx2word, word_freq
    )

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
    embedding_dim=512,
    window_size=5,
    num_negatives=5,
    batch_size=512,
    epochs=5,
    device="cuda",
    save_dir="../../models/word2vec/",
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

    target_emb = embeddings[word_idx].unsqueeze(0)
    similarities = F.cosine_similarity(target_emb, embeddings, dim=1)
    top_indices = similarities.topk(top_k + 1)[1][1:]

    similar_words = []
    for idx in top_indices:
        similar_word = idx2word[idx.item()]
        similarity_score = similarities[idx].item()
        similar_words.append((similar_word, similarity_score))

    return similar_words


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    tokens, counter = load_corpus(
        args.data_dir, max_sentences=args.max_sentences)
    word2idx, idx2word, word_freq = build_vocab(
        counter, min_count=args.min_count)
    token_ids = tokens_to_ids(tokens, word2idx)

    print(f"Vocabulary size: {len(word2idx)}")
    print(f"Total tokens: {len(token_ids):,}")
    print(f"Using device: {args.device}")

    vocab_path = os.path.join(args.save_dir, "vocab.pkl")
    save_vocab(word2idx, idx2word, word_freq, vocab_path)

    model = train_word2vec(
        token_ids=token_ids,
        word_freq=word_freq,
        vocab_size=len(word2idx),
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device,
        save_dir=args.save_dir,
        save_every=1,
    )

    test_words = ["the", "king", "good", "run"]
    for word in test_words:
        if word in word2idx:
            similar = find_similar_words(
                model, word, word2idx, idx2word, top_k=5, device=args.device
            )
            print(f"\nWords similar to '{word}':")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.4f}")


if __name__ == "__main__":
    download_and_extract_obw()
    main()

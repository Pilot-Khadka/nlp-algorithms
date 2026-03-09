import torch
import numpy as np

# pyrefly: ignore [missing-import]
import gensim.downloader as api
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_words(model, word, word2idx, idx2word, top_k=5, device="cpu"):
    if word not in word2idx:
        return []

    model.eval()
    word_id = torch.tensor([word2idx[word]]).to(device)
    emb = model.get_input_embeddings()[word_id].cpu().numpy()

    all_embs = model.get_input_embeddings().cpu().numpy()
    sims = cosine_similarity(emb, all_embs)[0]
    top_ids = sims.argsort()[::-1][1 : top_k + 1]
    return [(idx2word[i], sims[i]) for i in top_ids]


def load_gensim_model(model_name="word2vec-google-news-300"):
    return api.load(model_name)


def evaluate_similarity_dataset(model, word2idx, idx2word, dataset="wordsim353"):
    dataset = api.load(dataset)
    predicted = []
    actual = []

    embeddings = model.get_input_embeddings().cpu().numpy()
    for word1, word2, score in dataset:
        if word1 not in word2idx or word2 not in word2idx:
            continue

        idx1 = word2idx[word1]
        idx2 = word2idx[word2]

        vec1 = embeddings[idx1].reshape(1, -1)
        vec2 = embeddings[idx2].reshape(1, -1)

        sim = cosine_similarity(vec1, vec2)[0, 0]
        predicted.append(sim)
        actual.append(score)

    corr, _ = spearmanr(predicted, actual)
    print(
        f"""Word Similarity ({len(predicted)} pairs) - Spearman Correlation: {
            corr:.4f}"""
    )
    return corr


def evaluate_analogies(model, word2idx, idx2word, analogies):
    embeddings = model.get_input_embeddings().cpu().numpy()

    correct = 0
    total = 0

    for a, b, c, expected in analogies:
        if not all(w in word2idx for w in (a, b, c, expected)):
            continue

        vec = (
            embeddings[word2idx[b]] - embeddings[word2idx[a]] + embeddings[word2idx[c]]
        )
        norms = np.linalg.norm(embeddings, axis=1)
        sims = np.dot(embeddings, vec) / (norms * np.linalg.norm(vec))

        for ignore_idx in (word2idx[a], word2idx[b], word2idx[c]):
            sims[ignore_idx] = -np.inf  # mask input words

        pred_idx = np.argmax(sims)
        pred_word = idx2word[pred_idx]

        if pred_word == expected:
            correct += 1
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Analogy Accuracy: {correct}/{total} = {acc:.4f}")
    return acc


def compare_models(custom_model, custom_word2idx, custom_idx2word):
    gensim_model = load_gensim_model()

    analogies = [
        ("king", "man", "woman", "queen"),
        ("paris", "france", "italy", "rome"),
        ("walking", "walked", "running", "ran"),
        ("big", "bigger", "small", "smaller"),
        ("good", "better", "bad", "worse"),
    ]

    evaluate_analogies(custom_model, custom_word2idx, custom_idx2word, analogies)
    evaluate_similarity_dataset(custom_model, custom_word2idx, custom_idx2word)

    correct = 0
    total = 0
    for a, b, c, expected in analogies:
        try:
            result = gensim_model.most_similar(positive=[b, c], negative=[a], topn=1)
            if result[0][0] == expected:
                correct += 1
        except KeyError:
            continue
        total += 1

    acc = correct / total if total > 0 else 0
    print(f"Gensim Analogy Accuracy: {correct}/{total} = {acc:.4f}")

    sim_dataset = api.load("wordsim353")
    pred, actual = [], []
    for w1, w2, score in sim_dataset:
        try:
            sim = gensim_model.similarity(w1, w2)
            pred.append(sim)
            actual.append(score)
        except KeyError:
            continue
    corr, _ = spearmanr(pred, actual)
    print(f"Gensim WordSim353 Spearman Correlation: {corr:.4f}")

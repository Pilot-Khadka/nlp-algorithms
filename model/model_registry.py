import torch
import pickle

from core_vectorization.word2vec.word2vec import Word2Vec

MODEL_CLASS_MAP = {
    "word2vec": Word2Vec,
}


def load_vocab(filepath):
    with open(filepath, "rb") as f:
        vocab_data = pickle.load(f)
    print(f"Vocabulary loaded from {filepath}")
    return vocab_data


def load_model_from_name(filepath, device="cpu"):
    checkpoint = torch.load(filepath, map_location=device)
    class_name = checkpoint["model_class"]
    model_args = checkpoint["model_args"]

    if class_name not in MODEL_CLASS_MAP:
        raise ValueError(f"Unknown model class: {class_name}")

    ModelClass = MODEL_CLASS_MAP[class_name]
    model = ModelClass(**model_args)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print(f"Loaded model '{class_name}' from {filepath}")
    return model

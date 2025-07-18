import torch.nn as nn
from engine.registry import MODEL_REGISTRY


class BaseModel(nn.Module):
    def forward(self, x, hidden=None):
        raise NotImplementedError

    def init_hidden(self, batch_size, device):
        # override for models that need hidden states
        return None


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, model: BaseModel):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.model = model

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        return self.model(emb, hidden)


def create_model(model_type, vocab_size, embedding_dim, **model_kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' is not registered.")

    CoreModelClass = MODEL_REGISTRY[model_type]
    core_model = CoreModelClass(embedding_dim=embedding_dim, **model_kwargs)

    return LanguageModel(vocab_size, embedding_dim, core_model)

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


def create_model(cfg_model, dataset_bundle, cfg_task):
    model_type = cfg_model.name
    embedding_dim = cfg_model.embedding_dim
    print("task:", cfg_task)

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' is not registered.")

    CoreModelClass = MODEL_REGISTRY[model_type]
    model_kwargs = {k: v for k, v in cfg_model.items() if k not in ("name",)}

    output_dim = cfg_task.get_output_dim(dataset_bundle)
    core_model = CoreModelClass(output_dim=output_dim, **model_kwargs)
    return LanguageModel(dataset_bundle.vocab_size, embedding_dim, core_model)

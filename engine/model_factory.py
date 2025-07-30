import sys
import torch.nn as nn
from engine.registry import MODEL_REGISTRY


class BaseModel(nn.Module):
    def forward(self, x, hidden=None):
        raise NotImplementedError

    def init_hidden(self, batch_size, device):
        # override for models that need hidden states
        return None


def create_model(cfg_model, dataset_bundle, cfg_task, embedding_layer=None, vocab=None):
    task_name = cfg_task.name
    model_type = cfg_model.name

    if task_name == "language_modeling" and "bidirectional" in model_type.lower():
        print(
            f"""Bidirectional model '{model_type}' cannot be used with task '{
                task_name
            }'."""
        )
        sys.exit(0)

    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_type}' is not registered.")

    CoreModelClass = MODEL_REGISTRY[model_type]
    model_kwargs = {k: v for k, v in cfg_model.items() if k not in ("name",)}

    if cfg_task.name == "language_modeling":
        if cfg_model.use_pretrained_embedding:
            output_dim = len(vocab["word2idx"])
        else:
            output_dim = cfg_task.get_output_dim(dataset_bundle)
    else:
        output_dim = cfg_task.get_output_dim(dataset_bundle)

    core_model = CoreModelClass(
        output_dim=output_dim,
        embedding_layer=embedding_layer,
        **model_kwargs,
    )
    return core_model

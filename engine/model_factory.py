import torch.nn as nn
from typing import Any
from engine.registry import MODEL_REGISTRY


from models.model_registry import load_model_from_name


class BaseModel(nn.Module):
    def forward(self, x, hidden=None):
        raise NotImplementedError

    def init_hidden(self, batch_size, device) -> Any:
        return None


class ModelFactory:
    #  incompatible task-model combinations
    INCOMPATIBLE_COMBINATIONS = {"language_modeling": ["bidirectional"]}

    def __init__(self):
        self.model_registry = MODEL_REGISTRY

    def create_model(self, cfg_model, dataset_bundle, cfg_task) -> BaseModel:
        task_name = cfg_task.name
        model_type = cfg_model.name

        self._check_model_compatibility(task_name, model_type)
        embedding_layer = self._create_embeddings(dataset_bundle, cfg_model)
        output_dim = self._get_output_dim(cfg_task, cfg_model, dataset_bundle)

        core_model = self._create_core_model(
            model_type, cfg_model, embedding_layer, output_dim
        )

        return core_model

    def _check_model_compatibility(self, task_name: str, model_type: str) -> None:
        if task_name in self.INCOMPATIBLE_COMBINATIONS:
            incompatible_keywords = self.INCOMPATIBLE_COMBINATIONS[task_name]
            for keyword in incompatible_keywords:
                if keyword.lower() in model_type.lower():
                    raise ValueError(
                        f"Model '{model_type}' with '{keyword}' cannot be used "
                        f"with task '{task_name}'. Bidirectional models are not "
                        f"suitable for language modeling tasks."
                    )

    def _create_embeddings(self, dataset_bundle, cfg_model) -> nn.Embedding:
        vocab = dataset_bundle.vocab
        vocab_size = len(vocab)
        embedding_dim = cfg_model.embedding_dim

        use_pretrained = getattr(cfg_model, "use_pretrained_embedding", False)

        if use_pretrained:
            return self._create_pretrained_embeddings(
                cfg_model, vocab, vocab_size, embedding_dim
            )

        return self._create_random_embeddings(vocab_size, embedding_dim)

    def _create_pretrained_embeddings(
        self, cfg_model, vocab, vocab_size: int, embedding_dim: int
    ) -> nn.Embedding:
        if cfg_model.embedding_path is None:
            raise ValueError(
                "embedding_path is required when use_pretrained_embedding=True"
            )
        if cfg_model.vocab_path is None:
            raise ValueError(
                "vocab_path is required when use_pretrained_embedding=True"
            )

        w2v_model = load_model_from_name(cfg_model.embedding_path)
        pretrained_weights = w2v_model.get_input_embeddings().clone().detach().cpu()
        pretrained_vocab_size = pretrained_weights.shape[0]

        embedding_layer = nn.Embedding(vocab_size, embedding_dim)

        min_vocab_size = min(vocab_size, pretrained_vocab_size)
        embedding_layer.weight.data[:min_vocab_size] = pretrained_weights[
            :min_vocab_size
        ]

        if vocab_size > pretrained_vocab_size:
            nn.init.normal_(
                embedding_layer.weight.data[pretrained_vocab_size:], mean=0.0, std=0.1
            )

        return embedding_layer

    def _create_random_embeddings(
        self, vocab_size: int, embedding_dim: int
    ) -> nn.Embedding:
        embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        nn.init.normal_(embedding_layer.weight.data, mean=0.0, std=0.1)
        return embedding_layer

    def _get_output_dim(self, cfg_task, cfg_model, dataset_bundle) -> int:
        if cfg_task.name == "language_modeling":
            # for language modeling, output dim is vocabulary size
            return len(dataset_bundle.vocab)
        elif hasattr(cfg_task, "get_output_dim"):
            # for other tasks, use task-specific output dimension
            return cfg_task.get_output_dim(dataset_bundle)
        else:
            raise ValueError(
                f"Cannot determine output dimension for task '{cfg_task.name}'"
            )

    def _create_core_model(
        self, model_type: str, cfg_model, embedding_layer: nn.Embedding, output_dim: int
    ) -> BaseModel:
        if model_type not in self.model_registry:
            available_models = list(self.model_registry.keys())
            raise ValueError(
                f"Model '{model_type}' is not registered. "
                f"Available models: {available_models}"
            )

        CoreModelClass = self.model_registry[model_type]

        model_kwargs = {
            k: v
            for k, v in cfg_model.items()
            if k
            not in ("name", "use_pretrained_embedding", "embedding_path", "vocab_path")
        }

        core_model = CoreModelClass(
            output_dim=output_dim,
            embedding_layer=embedding_layer,
            **model_kwargs,
        )

        return core_model

    def list_available_models(self) -> list:
        return list(self.model_registry.keys())

    def is_model_available(self, model_type: str) -> bool:
        return model_type in self.model_registry

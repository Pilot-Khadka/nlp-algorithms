import warnings
import torch.nn as nn

from nlp_algorithms.engine.registry import (
    MODEL_REGISTRY,
    get_from_registry,
    TASK_REGISTRY,
)
from nlp_algorithms.engine.embedding_factory import EmbeddingFactory


class LanguageModel(nn.Module):
    def __init__(self, embedding, encoder, tie_weights=False):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder

        if not tie_weights:
            self.lm_head = nn.Linear(encoder.hidden_size, embedding.num_embeddings)
        else:
            # pyrefly: ignore [bad-assignment]
            self.lm_head = None

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        outputs, hidden = self.encoder(emb)

        if self.lm_head is None:
            return outputs, hidden

        logits = self.lm_head(outputs)
        return logits, hidden


class ClassificationModel(nn.Module):
    def __init__(self, embedding, encoder, num_classes):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder

        bidirectional = getattr(encoder, "bidirectional", False)
        hidden_dim = encoder.hidden_dim * 2 if bidirectional else encoder.hidden_dim

        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.bidirectional = bidirectional

    def forward(self, input_ids):
        emb = self.embedding(input_ids)
        outputs, hidden = self.encoder(emb)

        # outputs shape: (B, T, H) or (B, T, 2H) if bidirectional
        logits = self.output_layer(outputs)
        return logits, hidden


class ModelFactory:
    @staticmethod
    def create_model(
        model_config,
        dataset_bundle,
        task_config,
        data_config,
    ):
        task = get_from_registry(TASK_REGISTRY, task_config.name)
        all_flags = [k for k, v in model_config.items() if isinstance(v, bool) and v]
        model_variants = MODEL_REGISTRY[model_config.name].get("variants", {})

        variant_flag_names = set()
        for variant_key in model_variants.keys():
            variant_flag_names.update(variant_key)

        model_flags = [f for f in all_flags if f in variant_flag_names]
        other_flags = [f for f in all_flags if f not in variant_flag_names]

        if hasattr(task, "allowed_flags"):
            allowed = getattr(task, "allowed_flags")
            invalid_flags = set(other_flags) - set(allowed)
            if invalid_flags:
                raise ValueError(
                    f"Flags {invalid_flags} are not allowed for task {task_config.name}. "
                    f"Allowed flags: {allowed}"
                )
        elif other_flags:
            raise ValueError(
                f"Task {task_config.name} does not support any flags, "
                f"but received: {other_flags}"
            )

        if len(model_variants) > 1 and not model_flags:
            model_entry = MODEL_REGISTRY[model_config.name]
            if "default" in model_entry:
                pass
            elif frozenset({"unidirectional"}) in model_variants:
                model_flags = ["unidirectional"]
            else:
                raise ValueError(
                    f"Model '{model_config.name}' has multiple variants {[set(k) for k in model_variants.keys()]} "
                    f"but no flag was specified. Please provide one of the variant flags."
                )

        model_class = get_from_registry(
            MODEL_REGISTRY, model_config.name, flags=model_flags
        )

        embedding = EmbeddingFactory.create(model_config, dataset_bundle.vocab)
        output_dim = task.get_output_dim(dataset_bundle)

        # Variant flags drive class selection in the registry, so they are not
        # forwarded as constructor kwargs, the selected class already encodes that
        # behaviour (e.g. a bidirectional LSTM class rather than passing bidirectional=True).
        explicit_kwargs = {"input_dim": embedding.embedding_dim}

        encoder_kwargs = {
            k: v
            for k, v in model_config.items()
            if k not in variant_flag_names and k not in explicit_kwargs
        }

        encoder = model_class(**explicit_kwargs, **encoder_kwargs)

        if task_config.name == "language_modeling":
            return LanguageModel(
                embedding,
                encoder,
                tie_weights=model_config.weight_tying,
            )

        if task_config.name == "classification":
            num_classes = data_config.num_class
            return ClassificationModel(
                embedding=embedding,
                encoder=encoder,
                num_classes=num_classes,
            )

        raise ValueError(f"Unsupported task: {task_config.name}")

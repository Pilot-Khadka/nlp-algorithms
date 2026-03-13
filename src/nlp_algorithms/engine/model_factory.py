import torch
import torch.nn as nn
import torch.nn.functional as F

from nlp_algorithms.engine.registry import (
    MODEL_REGISTRY,
    get_from_registry,
    TASK_REGISTRY,
)
from nlp_algorithms.engine.embedding_factory import EmbeddingFactory


class EmbeddingDropout(nn.Module):
    def __init__(self, embedding: nn.Embedding, p: float = 0.1):
        super().__init__()
        self.embedding = embedding
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return self.embedding(x)
        vocab_size = self.embedding.weight.size(0)
        row_mask = self.embedding.weight.new_empty(vocab_size, 1).bernoulli_(
            1 - self.p
        ) / (1 - self.p)
        masked_weight = self.embedding.weight * row_mask
        return F.embedding(
            x,
            masked_weight,
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        )


class WordDropout(nn.Module):
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0:
            return x

        mask = x.new_empty(x.size(0), x.size(1), 1).bernoulli_(1 - self.p)
        return x * mask


class LanguageModel(nn.Module):
    def __init__(
        self,
        embedding,
        encoder,
        tie_weights: bool = False,
        embedding_dropout: float = 0.0,
        word_dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = embedding
        self.embedding_dropout = EmbeddingDropout(
            embedding=embedding, p=embedding_dropout
        )
        self.word_dropout = WordDropout(word_dropout)
        # lstm, rnn, gru, etc
        self.encoder = encoder
        hidden_size = encoder.hidden_dim
        vocab_size = embedding.num_embeddings
        emb_dim = embedding.embedding_dim
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_weights:
            if hidden_size != emb_dim:
                raise ValueError(
                    f"Cannot tie weights: hidden_size={hidden_size} "
                    f"!= embedding_dim={emb_dim}"
                )
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, *args, **kwargs):
        emb = self.word_dropout(self.embedding_dropout(input_ids))
        outputs, hidden = self.encoder(emb, *args, **kwargs)
        logits = self.lm_head(outputs)
        return logits, hidden


class ClassificationModel(nn.Module):
    def __init__(
        self,
        embedding,
        encoder,
        num_classes,
        embedding_dropout: float = 0.0,
        word_dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = embedding
        self.embedding_dropout = EmbeddingDropout(
            embedding=embedding, p=embedding_dropout
        )
        self.word_dropout = WordDropout(word_dropout)
        self.encoder = encoder

        bidirectional = getattr(encoder, "bidirectional", False)
        hidden_dim = encoder.hidden_dim * 2 if bidirectional else encoder.hidden_dim

        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.bidirectional = bidirectional

    def forward(self, input_ids):
        emb = self.word_dropout(self.embedding_dropout(input_ids))
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

        weight_tying = model_config.get("weight_tying", False)
        embedding_dropout = model_config.get("embedding_dropout", 0.0)
        word_dropout = model_config.get("embedding_word_dropout", 0.0)
        if task_config.name == "language_modeling":
            return LanguageModel(
                embedding=embedding,
                encoder=encoder,
                tie_weights=weight_tying,
                embedding_dropout=embedding_dropout,
                word_dropout=word_dropout,
            )

        if task_config.name == "classification":
            num_classes = data_config.num_class
            return ClassificationModel(
                embedding=embedding,
                encoder=encoder,
                num_classes=num_classes,
                embedding_dropout=embedding_dropout,
                word_dropout=word_dropout,
            )

        raise ValueError(f"Unsupported task: {task_config.name}")

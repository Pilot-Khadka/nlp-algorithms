from engine.registry import MODEL_REGISTRY, get_from_registry, TASK_REGISTRY
from engine.embedding_factory import EmbeddingFactory


class ModelFactory:
    @staticmethod
    def create_model(model_config, dataset_bundle, task_config):
        task = get_from_registry(TASK_REGISTRY, task_config.name)

        all_flags = [k for k, v in model_config.items() if isinstance(v, bool) and v]

        model_variants = MODEL_REGISTRY[model_config.name].get("variants", {})

        variant_flag_names = set()
        for variant_key in model_variants.keys():
            variant_flag_names.update(variant_key)  # frozenset unpacks to elements

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
            if frozenset({"unidirectional"}) in model_variants:
                model_flags = ["unidirectional"]
            else:
                default_variant = list(model_variants.keys())[0]
                model_flags = list(default_variant)
                import warnings

                warnings.warn(f"No variant flag specified, defaulting to {model_flags}")

        model_class = get_from_registry(
            MODEL_REGISTRY, model_config.name, flags=model_flags
        )

        embedding = EmbeddingFactory.create(model_config, dataset_bundle.vocab)
        output_dim = task.get_output_dim(dataset_bundle)

        model_kwargs = {
            k: v
            for k, v in model_config.items()
            if k
            not in (
                "name",
                "use_pretrained_embedding",
                "embedding_path",
                "vocab_path",
                "input_dim",
            )
            and k not in variant_flag_names
        }

        return model_class(
            input_dim=model_config.input_dim,
            output_dim=output_dim,
            embedding_layer=embedding,
            **model_kwargs,
        )

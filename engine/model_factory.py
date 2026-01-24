from engine.registry import MODEL_REGISTRY, get_from_registry, TASK_REGISTRY
from engine.embedding_factory import EmbeddingFactory


class ModelFactory:
    @staticmethod
    def create_model(model_config, dataset_bundle, task_config):
        task = get_from_registry(TASK_REGISTRY, task_config.name)

        flags = [k for k, v in model_config.items() if isinstance(v, bool) and v]

        if hasattr(task, "allowed_flags"):
            allowed = getattr(task, "allowed_flags")
            filtered_flags = [f for f in flags if f in allowed]
            invalid_flags = set(flags) - set(filtered_flags)
            if invalid_flags:
                raise ValueError(
                    f"Flags {invalid_flags} are not allowed for task {task_config.name}. "
                    f"Allowed flags: {allowed}"
                )
            flags = filtered_flags

        model_class = get_from_registry(MODEL_REGISTRY, model_config.name, flags=flags)
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
        }

        return model_class(
            input_dim=model_config.input_dim,
            output_dim=output_dim,
            embedding_layer=embedding,
            **model_kwargs,
        )

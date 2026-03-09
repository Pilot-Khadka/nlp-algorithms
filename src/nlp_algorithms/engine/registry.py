from typing import Dict, Any, Optional, FrozenSet, Callable
import importlib
import pkgutil

from nlp_algorithms.util.path_util import get_project_base_path

_REGISTRY: Dict[str, Any] = {}
RegistryEntry = Dict[str, Any]

# def _create_register(category: str) -> Callable:
#     def register(name: str, *, flags=None):
#         flags = frozenset(flags or [])
#
#         def decorator(obj):
#             _REGISTRY.setdefault(category, {}).setdefault(name, {})[flags] = obj
#             return obj
#
#         return decorator
#
#     return register


def _create_register(category: str) -> Callable:
    def register(name: str, *, flags=None):
        flags = frozenset(flags or [])

        def decorator(obj):
            entry = _REGISTRY.setdefault(category, {}).setdefault(name, {})
            if not flags:
                entry["default"] = obj
            else:
                entry.setdefault("variants", {})[flags] = obj
            return obj

        return decorator

    return register


MODEL_REGISTRY = _REGISTRY.setdefault("model", {})
DATA_READER_REGISTRY = _REGISTRY.setdefault("data_reader", {})
COLLATOR_REGISTRY = _REGISTRY.setdefault("collator", {})
TOKENIZER_REGISTRY = _REGISTRY.setdefault("tokenizer", {})
VECTORIZER_REGISTRY = _REGISTRY.setdefault("vectorizer", {})
TASK_REGISTRY = _REGISTRY.setdefault("task", {})
TRAINER_REGISTRY = _REGISTRY.setdefault("trainer", {})
DOWNLOADER_REGISTRY = _REGISTRY.setdefault("downloader", {})


register_model = _create_register("model")
register_reader = _create_register("data_reader")
register_collator = _create_register("collator")
register_tokenizer = _create_register("tokenizer")
register_vectorizer = _create_register("vectorizer")
register_task = _create_register("task")
register_trainer = _create_register("trainer")
register_downloader = _create_register("downloader")


def autoregister():
    """used to discover registerable files (do not delete)"""
    import sys

    root = get_project_base_path()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for module in pkgutil.walk_packages(path=[str(root)]):
        if module.ispkg:
            continue  # skip packages, only import .py files
        try:
            importlib.import_module(module.name)
        except Exception as e:
            print(f"[WARN] Skipped {module.name}: {e}")


def get_from_registry(
    registry: Dict[str, RegistryEntry],
    name: str,
    flags: Optional[list[str]] = None,
) -> Any:
    """Get a class/function from any registry by name and optional flags."""
    entry = registry.get(name)
    if entry is None:
        raise KeyError(f"{name} not found in registry")

    flags_set = frozenset(flags or [])

    # if no flags, try 'default' first
    if not flags_set:
        if "default" in entry:
            return entry["default"]
        # fallback to variants
        variants = entry.get("variants", {})
        if len(variants) == 1:
            return next(iter(variants.values()))
        raise KeyError(
            f"Multiple variants found for {name}; flags required. Available: {list(variants.keys())}"
        )

    # if flags provided, look up inside 'variants'
    obj = entry.get("variants", {}).get(flags_set)
    if obj is None:
        raise KeyError(
            f"No variant found for {name} with flags {flags_set}. "
            f"Available: {list(entry.get('variants', {}).keys())}"
        )
    return obj

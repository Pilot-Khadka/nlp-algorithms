from typing import Dict

MODEL_REGISTRY: Dict = {}


def auto_register_models():
    import sys
    import importlib
    import pkgutil
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    for module in pkgutil.walk_packages(path=[str(root)]):
        if module.ispkg:
            continue  # skip packages, only import .py files
        try:
            importlib.import_module(module.name)
        except Exception as e:
            print(f"[WARN] Skipped {module.name}: {e}")


def register_model(name, *flags):
    flags = frozenset(flags)

    def decorator(cls):
        variants = MODEL_REGISTRY.setdefault(name, {})
        if flags in variants:
            raise ValueError(
                f"Duplicate registration for model '{name}' with flags {flags}"
            )
        variants[flags] = cls
        return cls

    return decorator

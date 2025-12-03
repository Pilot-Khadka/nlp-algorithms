import os
import sys
import yaml
import importlib
from pathlib import Path

MODEL_REGISTRY = {}

CONFIG_ROOT = "conf"
MODEL_DIR = os.path.join(CONFIG_ROOT, "model")
TRAINING_DIR = os.path.join(CONFIG_ROOT, "training")


def auto_register_models():
    root = Path(__file__).resolve().parent.parent

    sys.path.insert(0, str(root))

    for item in sorted(root.iterdir()):
        if item.is_dir() and item.name.startswith("a") and "_" in item.name:
            for py_file in item.rglob("*.py"):
                try:
                    with open(py_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    if "__register_model__ = True" not in content:
                        continue

                    rel_path = py_file.relative_to(root)
                    module_parts = rel_path.with_suffix("").parts
                    module_name = ".".join(module_parts)

                    importlib.import_module(module_name)

                except Exception as e:
                    print(f"[WARN] Skipped {py_file}: {e}")


def _write_yaml(path, payload):
    # Writes only when the file does not exist to avoid silent overrides.
    if not os.path.exists(path):
        with open(path, "w") as f:
            yaml.dump(payload, f, sort_keys=False)


def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls

        model_cfg_path = os.path.join(MODEL_DIR, f"{name}.yaml")
        train_cfg_path = os.path.join(TRAINING_DIR, f"{name}.yaml")

        _write_yaml(
            model_cfg_path,
            {"name": name},
        )

        _write_yaml(
            train_cfg_path,
            {
                "training": {
                    "optimizer": "adam",
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 10,
                }
            },
        )

        return cls

    return decorator

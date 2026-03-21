from pathlib import Path

import nlp_algorithms


def get_project_base_path() -> Path:
    current = Path(nlp_algorithms.__file__).resolve()

    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent.resolve()

    raise RuntimeError("Could not determine project base path.")


def get_data_path() -> Path:
    return (get_project_base_path() / "data").resolve()

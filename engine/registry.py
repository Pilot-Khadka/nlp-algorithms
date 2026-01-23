from typing import Dict


MODEL_REGISTRY: Dict = {}
DATASET_READER: Dict = {}
COLLATOR: Dict = {}
TOKENIZER: Dict = {}
VECTORIZER: Dict = {}
TASK: Dict = {}
TRAINER_REGISTRY: Dict = {}
DOWNLOADER: Dict = {}


def register_model(name, *, flags=None):
    """flags define model capabilities"""
    flags = frozenset(flags or [])

    def decorator(cls):
        MODEL_REGISTRY.setdefault(name, {})[flags] = cls
        return cls

    return decorator


def register_dataset(name):
    def wrapper(cls):
        DATASET_READER[name] = cls
        return cls

    return wrapper


def register_collator(task):
    def wrapper(cls):
        COLLATOR[task] = cls
        return cls

    return wrapper


def register_tokenizer(name):
    def wrapper(cls):
        TOKENIZER[name] = cls
        return cls

    return wrapper


def register_task(name):
    def wrapper(cls):
        TASK[name] = cls

    return wrapper


def register_vectorizer(name):
    def wrapper(cls):
        VECTORIZER[name] = cls

    return wrapper


def register_trainer(name):
    def wrapper(cls):
        TRAINER_REGISTRY[name] = cls

    return wrapper


def register_downloader(name):
    def wrapper(cls):
        DOWNLOADER[name] = cls

    return wrapper

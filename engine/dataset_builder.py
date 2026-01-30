from tqdm import tqdm
from collections import Counter


from engine.registry import (
    get_from_registry,
    DATA_READER_REGISTRY,
    DOWNLOADER_REGISTRY,
    TOKENIZER_REGISTRY,
)
from infra.vocabulary import Vocabulary


class DatasetBundle:
    def __init__(
        self,
        train_dataset,
        val_dataset,
        test_dataset,
        token_vocab,
        label_vocab=None,
        src_vocab=None,
        tgt_vocab=None,
        tokenizer=None,
    ):
        self.train = train_dataset
        self.val = val_dataset
        self.test = test_dataset
        self.vocab = token_vocab
        self.label_vocab = label_vocab
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tokenizer = tokenizer


def build_vocab_from_key(
    dataset,
    key: str,
    tokenizer=None,
    vocab_size: int = 10000,
    min_freq: int = 1,
    special_tokens=None,
):
    if special_tokens is None:
        special_tokens = {"<pad>": 0, "<unk>": 1}

    counter = Counter()

    for item in tqdm(dataset, desc=f"Building vocab from '{key}'"):
        val = item[key]

        if tokenizer is not None:
            # text field
            if isinstance(val, str):
                counter.update(tokenizer.tokenize(val))
            elif isinstance(val, list):
                for text in val:
                    counter.update(tokenizer.tokenize(text))
            else:
                raise ValueError(f"Unsupported type for key '{key}': {type(val)}")
        else:
            # label field
            if isinstance(val, list):
                counter.update(val)
            else:
                counter.update([val])

    # build token_to_id
    token_to_id = dict(special_tokens)
    idx = len(token_to_id)

    for token, freq in counter.most_common(vocab_size):
        if freq >= min_freq and token not in token_to_id:
            token_to_id[token] = idx
            idx += 1

    return Vocabulary(token_to_id)


class DatasetBundleBuilder:
    def build(self, config):
        data_downloader_cls = get_from_registry(
            DOWNLOADER_REGISTRY, config.dataset.name
        )
        data_dir = data_downloader_cls().download_and_prepare(config)

        data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name)
        train = data_reader_cls(data_dir=data_dir, split="train")
        test = data_reader_cls(data_dir=data_dir, split="test")

        tokenizer_cls = get_from_registry(TOKENIZER_REGISTRY, config.tokenizer.name)
        tokenizer = tokenizer_cls()

        label_vocab = None
        token_vocab = None
        src_vocab = None
        tgt_vocab = None

        if config.task.name in {"classification", "ner"}:
            label_vocab = build_vocab_from_key(
                train, key="label", tokenizer=None, vocab_size=config.dataset.vocab_size
            )
        elif config.task.name == "translation":
            src_vocab = build_vocab_from_key(
                train,
                key="src",
                tokenizer=tokenizer,
                vocab_size=config.dataset.vocab_size,
            )
            tgt_vocab = build_vocab_from_key(
                train,
                key="tgt",
                tokenizer=tokenizer,
                vocab_size=config.dataset.vocab_size,
            )
        else:
            token_vocab = build_vocab_from_key(
                train,
                key="text",
                tokenizer=tokenizer,
                vocab_size=config.dataset.vocab_size,
            )

        return DatasetBundle(
            train_dataset=train,
            val_dataset=test,
            test_dataset=test,
            token_vocab=token_vocab,
            label_vocab=label_vocab,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
        )

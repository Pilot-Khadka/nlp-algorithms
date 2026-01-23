from engine.registry import DATASET_READER, DOWNLOADER, TOKENIZER
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


def build_token_vocab(train, tokenizer):
    all_tokens = []
    for item in train:
        all_tokens.extend(tokenizer.tokenize(item["text"]))

    vocab = Vocabulary.from_tokens(tokens=all_tokens)
    return vocab


def build_label_vocab(dataset, key="label"):
    """For Classification or NER"""
    all_labels = []
    for item in dataset:
        val = item[key]
        if isinstance(val, list):  # NER case
            all_labels.extend(val)
        else:  # classification
            all_labels.append(val)
    return Vocabulary.from_tokens(all_labels)


def build_token_vocab_from_key(train, tokenizer, key):
    pass


class DatasetBundleBuilder:
    def build(self, config):
        data_downloader = DOWNLOADER[config.dataset.name]
        data_downloader.download_and_prepare(config)

        data_reader = DATASET_READER[config.dataset.name]
        train = data_reader(
            data_dir=config.dataset.data_dir,
            split="train",
        )
        val = data_reader(
            data_dir=config.dataset.data_dir,
            split="val",
        )
        test = data_reader(
            data_dir=config.dataset.data_dir,
            split="test",
        )

        tokenizer = TOKENIZER[config.dataset.tokenizer.name]
        token_vocab = build_token_vocab(train, tokenizer)

        label_vocab = None
        src_vocab = None
        tgt_vocab = None

        if config.task.name in {"classification", "ner"}:
            label_vocab = build_label_vocab(train, key="label")

        elif config.task.name == "translation":
            # For MT, text is in "src" and "tgt" columns
            src_vocab = build_token_vocab_from_key(train, tokenizer, key="src")
            tgt_vocab = build_token_vocab_from_key(train, tokenizer, key="tgt")

        return DatasetBundle(
            train_dataset=train,
            val_dataset=val,
            test_dataset=test,
            token_vocab=token_vocab,
            label_vocab=label_vocab,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            tokenizer=tokenizer,
        )

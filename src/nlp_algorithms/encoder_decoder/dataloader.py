import inspect

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

from nlp_algorithms.util.general_util import get_num_workers
from nlp_algorithms.engine.dataset_builder import build_vocab_from_key
from nlp_algorithms.infra.preprocessor import PreprocessedDataset
from nlp_algorithms.engine.registry import (
    DATA_READER_REGISTRY,
    DOWNLOADER_REGISTRY,
    COLLATOR_REGISTRY,
    TOKENIZER_REGISTRY,
    get_from_registry,
)
from nlp_algorithms.util.multi_gpu import is_rank0


class DatasetBundle:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        train_sampler,
        test_sampler,
        label_vocab,
        src_vocab,
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.label_vocab = label_vocab
        self.src_vocab = src_vocab
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler


def init_tokenizer(tokenizer_name, config):
    tokenizer_cls = get_from_registry(TOKENIZER_REGISTRY, tokenizer_name)
    tokenizer_kwargs = {}
    sig = inspect.signature(tokenizer_cls.__init__)

    if "vocab_size" in sig.parameters:
        tokenizer_kwargs["vocab_size"] = getattr(
            config.tokenizer, "vocab_size", config.dataset.vocab_size
        )

    return tokenizer_cls(**tokenizer_kwargs)


def get_dataloaders_distributed(config, world_size, rank):
    data_downloader_cls = get_from_registry(DOWNLOADER_REGISTRY, config.dataset.name)

    # pyrefly: ignore [bad-argument-count]
    if is_rank0(rank):
        data_dir = data_downloader_cls().download_and_prepare(config)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    data_dir = config.dataset.data_dir

    if config.dataset.name == "huggingface":
        data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name2)
    else:
        data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name)

    max_samples = config.dataset.get("max_samples", None)
    train = data_reader_cls(
        data_dir=data_dir,
        split="train",
        max_samples=max_samples,
    )
    test = data_reader_cls(
        data_dir=data_dir,
        split="test",
        max_samples=max_samples,
    )

    tokenizer_src = init_tokenizer(config.tokenizer.name, config)
    tokenizer_tgt = init_tokenizer(config.tokenizer.name, config)

    src_vocab = build_vocab_from_key(
        dataset=train,
        config=config,
        tokenizer=tokenizer_src,
        key="src",
    )
    tgt_vocab = build_vocab_from_key(
        dataset=train,
        config=config,
        tokenizer=tokenizer_tgt,
        key="tgt",
    )

    processed_train = PreprocessedDataset(
        train,
        src_tokenizer=tokenizer_src,
        tgt_tokenizer=tokenizer_tgt,
        vocab=src_vocab,
        target_vocab=tgt_vocab,
        max_len=config.dataset.sequence_length,
        task=config.task.name,
    )
    processed_test = PreprocessedDataset(
        test,
        src_tokenizer=tokenizer_src,
        tgt_tokenizer=tokenizer_tgt,
        vocab=src_vocab,
        target_vocab=tgt_vocab,
        max_len=config.dataset.sequence_length,
        task=config.task.name,
    )
    label_vocab = tgt_vocab

    collator = get_from_registry(COLLATOR_REGISTRY, config.task.name)()
    num_workers = config.dataset.get("num_workers", "auto")
    num_workers = get_num_workers(num_workers=num_workers)
    pin_memory = config.dataset.get("pin_memory", True)
    prefetch_factor = config.dataset.get("prefetch_factor", 2)

    if world_size > 1:
        train_sampler = DistributedSampler(processed_train, shuffle=True)
        test_sampler = DistributedSampler(processed_test, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        processed_train,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        collate_fn=collator.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        processed_test,
        batch_size=config.train.batch_size,
        sampler=test_sampler,
        collate_fn=collator.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    bundle = DatasetBundle(
        train_loader=train_loader,
        valid_loader=test_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
        src_vocab=src_vocab,
        label_vocab=label_vocab,
    )
    return bundle

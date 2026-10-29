import os
import re
import random
import unicodedata
import functools
import multiprocessing
from os.path import exists
from pathlib import Path

import torch
import numpy as np
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict
from datasets import Dataset as HFDataset

from torch.utils.data import Dataset, Sampler
from torch.nn.functional import pad
from torch.utils.data import DataLoader, DistributedSampler

from nlp_algorithms.tokenization import BytePairEncoder

_SPECIALS = ["<s>", "</s>", "<blank>", "<unk>"]
_SPECIAL_OFFSET = len(_SPECIALS)

BPE_VOCAB_SIZE = 8192


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        return self.hf_dataset[index]


def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


class Batch:
    def __init__(self, src, tgt=None, pad=2):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def data_gen(V, batch_size, nbatches):
    for _ in range(nbatches):
        data = torch.randint(1, V, size=(batch_size, 10))
        data[:, 0] = 1
        src = data.requires_grad_(False).clone().detach()
        tgt = data.requires_grad_(False).clone().detach()
        yield Batch(src, tgt, 0)


class Vocab:
    """
    wraps bpe encoder with 4 special tokens the seq2seq model expects

    bpe assigns 0-255 to raw bytes, 256+ to merges
    seq2seq reserves 0:<s>, 1:</s>, 2:<blank> and 3:<unk>
    to avoid collision, bpe id is shifted by _SPECIAL_OFFSET=4
    """

    def __init__(self, bpe: BytePairEncoder):
        self.bpe = bpe
        self._itos = list(_SPECIALS) + [
            bpe.vocab[i].decode("utf-8", errors="replace")
            for i in sorted(bpe.vocab.keys())
        ]
        self._stoi = {tok: i for i, tok in enumerate(_SPECIALS)}

    def __len__(self):
        return len(self._itos)

    def __getitem__(self, token):
        return self._stoi.get(token, _SPECIALS.index("<unk>"))

    def get_itos(self):
        return self._itos

    def encode(self, text: str) -> list[int]:
        return [i + _SPECIAL_OFFSET for i in self.bpe.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        bpe_ids = [i - _SPECIAL_OFFSET for i in ids if i >= _SPECIAL_OFFSET]
        return self.bpe.detokenize(bpe_ids)


def load_multi30k(src_lang="de", tgt_lang="en"):
    data_path = Path("data/multi30k/bentrevett___multi30k")

    if data_path.exists():
        dataset = load_from_disk(str(data_path))
    else:
        dataset = load_dataset("bentrevett/multi30k")
        dataset = dataset.map(
            lambda row: {"src": row[src_lang], "tgt": row[tgt_lang]},
            remove_columns=dataset["train"].column_names,
        )
        data_path.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(data_path))

    return dataset["train"], dataset["validation"], dataset["test"]


def load_tatoeba(data_dir, url):
    # tatoeba challenge release: {train,dev,test}.{src,trg}, one sentence per line
    from nlp_algorithms.dataset.downloader.tatoeba import TatoebaDownloader

    TatoebaDownloader.download_and_prepare(
        {"dataset": {"data_dir": data_dir, "url": url}}
    )

    def read_lines(path):
        # only \n ends a line, a stray \r would shift src against tgt
        with open(path, encoding="utf-8", newline="\n") as f:
            return [line.rstrip("\n") for line in f]

    def read_split(split):
        src = read_lines(os.path.join(data_dir, f"{split}.src"))
        tgt = read_lines(os.path.join(data_dir, f"{split}.trg"))
        if len(src) != len(tgt):
            raise ValueError(f"{split}: {len(src)} src lines but {len(tgt)} tgt lines")
        return HFDataset.from_dict({"src": src, "tgt": tgt})

    has_dev = os.path.exists(os.path.join(data_dir, "dev.src"))
    return (
        read_split("train"),
        read_split("dev" if has_dev else "test"),
        read_split("test"),
    )


_ZERO_WIDTH = re.compile("[​‌‍﻿]")
_DEVA_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def normalize_ne(text):
    # flores writes numbers with ascii digits and the danda without a space
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text).translate(_DEVA_DIGITS)
    text = re.sub(r"\s+([।॥])", r"\1", text)
    return " ".join(text.split())


def normalize_en(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def clean_pairs(src, tgt, max_ratio=3.0):
    # drops empty, untranslated, duplicate and badly aligned pairs
    seen = set()
    out_src, out_tgt = [], []
    for s, t in zip(src, tgt):
        if not s or not t or s == t or (s, t) in seen:
            continue
        if not 1 / max_ratio <= len(s) / len(t) <= max_ratio:
            continue
        seen.add((s, t))
        out_src.append(s)
        out_tgt.append(t)
    return out_src, out_tgt


def _normalize_ne_en(train, val, test):
    # in memory, dataset.map would try to write a cache next to read-only kaggle input
    src, tgt = clean_pairs(
        [normalize_ne(s) for s in train["src"]],
        [normalize_en(t) for t in train["tgt"]],
    )
    print(f"ne_en train: {len(train)} -> {len(src)} pairs after cleaning")
    train = HFDataset.from_dict({"src": src, "tgt": tgt})
    # the test references stay as they are for bleu
    val = HFDataset.from_dict(
        {
            "src": [normalize_ne(s) for s in val["src"]],
            "tgt": [normalize_en(t) for t in val["tgt"]],
        }
    )
    test = HFDataset.from_dict(
        {"src": [normalize_ne(s) for s in test["src"]], "tgt": list(test["tgt"])}
    )
    return train, val, test


def load_ne_en(src_lang_col="ne", tgt_lang_col="en", normalize=True):
    data_path = Path("data/ne_en")

    if data_path.exists():
        dataset = load_from_disk(str(data_path))
        splits = dataset["train"], dataset["validation"], dataset["test"]
        return _normalize_ne_en(*splits) if normalize else splits

    train_raw = load_dataset("sharad461/ne-en-parallel-208k", split="train")
    train_data = train_raw.map(
        lambda row: {"src": row[src_lang_col], "tgt": row[tgt_lang_col]},
        remove_columns=train_raw.column_names,
    )

    def load_flores_split(ne_file, en_file):
        ne = load_dataset(
            "openlanguagedata/flores_plus", data_files=ne_file, split="train"
        )
        en = load_dataset(
            "openlanguagedata/flores_plus", data_files=en_file, split="train"
        )
        return HFDataset.from_dict(
            {
                "src": ne["text"],
                "tgt": en["text"],
            }
        )

    val_data = load_flores_split("dev/npi_Deva.jsonl", "dev/eng_Latn.jsonl")
    test_data = load_flores_split("devtest/npi_Deva.jsonl", "devtest/eng_Latn.jsonl")
    dataset = DatasetDict(
        {
            "train": train_data,
            "validation": val_data,
            "test": test_data,
        }
    )
    data_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(data_path))
    splits = dataset["train"], dataset["validation"], dataset["test"]
    return _normalize_ne_en(*splits) if normalize else splits


def build_vocabulary(load_fn, vocab_size=BPE_VOCAB_SIZE, joint=False, sample_size=None):
    train, val, _ = load_fn()
    if joint:
        # one bpe for both languages so the embeddings can be shared,
        # names and numbers get the same ids on both sides
        rows = list(zip(train["src"], train["tgt"]))
        if sample_size is not None and sample_size < len(rows):
            # bpe training keeps python objects per byte, the full corpus does not fit in memory
            rows = random.Random(42).sample(rows, sample_size)
        print("Training joint BPE tokenizer...")
        bpe = BytePairEncoder(vocab_size=vocab_size)
        bpe.train("\n".join(text for row in rows for text in row))
        vocab = Vocab(bpe)
        return vocab, vocab

    src_text = "\n".join(list(train["src"]) + list(val["src"]))
    tgt_text = "\n".join(list(train["tgt"]) + list(val["tgt"]))

    print("Training Source BPE tokenizer...")
    bpe_src = BytePairEncoder(vocab_size=vocab_size)
    bpe_src.train(src_text)

    print("Training Target BPE tokenizer...")
    bpe_tgt = BytePairEncoder(vocab_size=vocab_size)
    bpe_tgt.train(tgt_text)
    return Vocab(bpe_src), Vocab(bpe_tgt)


def load_vocab(load_fn, vocab_path, vocab_size=BPE_VOCAB_SIZE, **kwargs):
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(load_fn, vocab_size, **kwargs)
        os.makedirs(os.path.dirname(vocab_path) or ".", exist_ok=True)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path, weights_only=False)
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def pretokenize(rows, vocab_src, vocab_tgt, num_workers=1, show_progress=True):
    # mininterval keeps the progress bar from flooding non-terminal logs (kaggle)
    if num_workers > 1:
        # bpe is pure python, split the rows over processes
        rows = list(rows)
        size = -(-len(rows) // (num_workers * 20))
        chunks = [rows[i : i + size] for i in range(0, len(rows), size)]
        fn = functools.partial(
            pretokenize, vocab_src=vocab_src, vocab_tgt=vocab_tgt, show_progress=False
        )
        with multiprocessing.get_context("fork").Pool(num_workers) as pool:
            done = tqdm(
                pool.imap(fn, chunks),
                total=len(chunks),
                desc="tokenizing",
                mininterval=60,
            )
            return [row for chunk in done for row in chunk]

    return [
        {
            "src_ids": vocab_src.encode(row["src"]),
            "tgt_ids": vocab_tgt.encode(row["tgt"]),
        }
        for row in tqdm(
            rows, desc="tokenizing", mininterval=60, disable=not show_progress
        )
    ]


def collate_batch(
    batch,
    vocab_src,
    vocab_tgt,
    device,
    max_padding=512,
    pad_id=2,
):
    # built on cpu and moved to the device once per batch
    bs_id = torch.tensor([0])
    eos_id = torch.tensor([1])
    src_list, tgt_list = [], []

    max_content_len = max_padding - 2  # reserve slots for BOS + EOS

    for sample in batch:
        # samples from pretokenize() already have ids
        if "src_ids" in sample:
            src_ids, tgt_ids = sample["src_ids"], sample["tgt_ids"]
        else:
            src_ids = vocab_src.encode(sample["src"])
            tgt_ids = vocab_tgt.encode(sample["tgt"])

        src_ids = torch.tensor(src_ids, dtype=torch.int64)[:max_content_len]
        tgt_ids = torch.tensor(tgt_ids, dtype=torch.int64)[:max_content_len]

        src_list.append(torch.cat([bs_id, src_ids, eos_id], 0))
        tgt_list.append(torch.cat([bs_id, tgt_ids, eos_id], 0))

    # pad to the longest sentence in the batch instead of max_padding,
    # padded positions are masked so the loss is the same
    src_len = max(len(x) for x in src_list)
    tgt_len = max(len(x) for x in tgt_list)
    src = torch.stack([pad(x, (0, src_len - len(x)), value=pad_id) for x in src_list])
    tgt = torch.stack([pad(x, (0, tgt_len - len(x)), value=pad_id) for x in tgt_list])
    return src.to(device), tgt.to(device)


class TokenBatchSampler(Sampler):
    """
    batches of similar length sentences with at most max_tokens (padded) tokens,
    like fairseq --max-tokens instead of a fixed number of sentences
    """

    def __init__(self, lengths, max_tokens, shuffle=True, num_replicas=1, rank=0):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.batches = self._make_batches()

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.batches = self._make_batches()

    def _make_batches(self):
        rng = random.Random(42 + self.epoch)
        order = list(range(len(self.lengths)))
        if self.shuffle:
            rng.shuffle(order)
        # stable sort, equal lengths keep the shuffled order
        order.sort(key=lambda i: self.lengths[i])

        batches, batch, longest = [], [], 0
        for i in order:
            longest_if_added = max(longest, self.lengths[i])
            if batch and longest_if_added * (len(batch) + 1) > self.max_tokens:
                batches.append(batch)
                batch, longest_if_added = [], self.lengths[i]
            batch.append(i)
            longest = longest_if_added
        if batch:
            batches.append(batch)

        if self.shuffle:
            rng.shuffle(batches)
            # every gpu needs the same number of steps or ddp hangs
            batches = batches[: len(batches) // self.num_replicas * self.num_replicas]
        return batches[self.rank :: self.num_replicas]

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    load_fn,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
    max_tokens=None,
):
    def collate_fn(batch):
        return collate_batch(
            batch,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src["<blank>"],
        )

    train_data, valid_data, _ = load_fn()
    if max_tokens is not None:
        return _token_dataloaders(
            train_data, valid_data, collate_fn, max_tokens, max_padding, is_distributed
        )

    train_data = HFDatasetWrapper(train_data)
    valid_data = HFDatasetWrapper(valid_data)

    train_sampler = DistributedSampler(train_data) if is_distributed else None
    valid_sampler = DistributedSampler(valid_data) if is_distributed else None

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )
    return train_dataloader, valid_dataloader


def _token_dataloaders(
    train_data, valid_data, collate_fn, max_tokens, max_padding, is_distributed
):
    # needs pretokenized rows to know the lengths up front
    def lengths(rows):
        return [
            min(max(len(r["src_ids"]), len(r["tgt_ids"])) + 2, max_padding)
            for r in rows
        ]

    num_replicas, rank = 1, 0
    if is_distributed:
        num_replicas, rank = dist.get_world_size(), dist.get_rank()

    train_sampler = TokenBatchSampler(
        lengths(train_data), max_tokens, num_replicas=num_replicas, rank=rank
    )
    # every gpu evaluates the whole validation set, it is small
    valid_sampler = TokenBatchSampler(lengths(valid_data), max_tokens, shuffle=False)

    train_dataloader = DataLoader(
        train_data, batch_sampler=train_sampler, collate_fn=collate_fn
    )
    valid_dataloader = DataLoader(
        valid_data, batch_sampler=valid_sampler, collate_fn=collate_fn
    )
    return train_dataloader, valid_dataloader

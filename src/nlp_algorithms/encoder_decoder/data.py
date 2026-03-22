from os.path import exists
from pathlib import Path

import torch
import numpy as np
from datasets import load_dataset, load_from_disk, DatasetDict
from datasets import Dataset as HFDataset

from torch.utils.data import Dataset
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


def load_ne_en(src_lang_col="ne", tgt_lang_col="en"):
    data_path = Path("data/ne_en")

    if data_path.exists():
        dataset = load_from_disk(str(data_path))
        return dataset["train"], dataset["validation"], dataset["test"]

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
    return dataset["train"], dataset["validation"], dataset["test"]


def build_vocabulary(load_fn, vocab_size=BPE_VOCAB_SIZE):
    train, val, _ = load_fn()
    src_text = "\n".join(list(train["src"]) + list(val["src"]))
    tgt_text = "\n".join(list(train["tgt"]) + list(val["tgt"]))

    print("Training Source BPE tokenizer...")
    bpe_src = BytePairEncoder(vocab_size=vocab_size)
    bpe_src.train(src_text)

    print("Training Target BPE tokenizer...")
    bpe_tgt = BytePairEncoder(vocab_size=vocab_size)
    bpe_tgt.train(tgt_text)
    return Vocab(bpe_src), Vocab(bpe_tgt)


def load_vocab(load_fn, vocab_path="vocab.pt", vocab_size=BPE_VOCAB_SIZE):
    if not exists(vocab_path):
        vocab_src, vocab_tgt = build_vocabulary(load_fn, vocab_size)
        torch.save((vocab_src, vocab_tgt), vocab_path)
    else:
        vocab_src, vocab_tgt = torch.load(vocab_path, weights_only=False)
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def collate_batch(
    batch,
    vocab_src,
    vocab_tgt,
    device,
    max_padding=128,
    pad_id=2,
):
    bs_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)
    src_list, tgt_list = [], []

    for sample in batch:
        src_ids = torch.tensor(
            vocab_src.encode(sample["src"]), dtype=torch.int64, device=device
        )
        tgt_ids = torch.tensor(
            vocab_tgt.encode(sample["tgt"]), dtype=torch.int64, device=device
        )

        processed_src = torch.cat([bs_id, src_ids, eos_id], 0)
        processed_tgt = torch.cat([bs_id, tgt_ids, eos_id], 0)

        src_list.append(
            pad(processed_src, (0, max_padding - len(processed_src)), value=pad_id)
        )
        tgt_list.append(
            pad(processed_tgt, (0, max_padding - len(processed_tgt)), value=pad_id)
        )

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    load_fn,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
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

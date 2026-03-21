from os.path import exists

import torch
import numpy as np
from datasets import load_dataset

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


def load_multi30k():
    dataset = load_dataset("bentrevett/multi30k")
    return dataset["train"], dataset["validation"], dataset["test"]


def build_vocabulary(vocab_size=BPE_VOCAB_SIZE):
    train, val, _ = load_multi30k()

    de_text = "\n".join(list(train["de"]) + list(val["de"]))
    en_text = "\n".join(list(train["en"]) + list(val["en"]))

    print("Training German BPE tokenizer...")
    bpe_src = BytePairEncoder(vocab_size=vocab_size)
    bpe_src.train(de_text)

    print("Training English BPE tokenizer...")
    bpe_tgt = BytePairEncoder(vocab_size=vocab_size)
    bpe_tgt.train(en_text)

    return Vocab(bpe_src), Vocab(bpe_tgt)


def load_vocab(vocab_size=BPE_VOCAB_SIZE):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(vocab_size)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt", weights_only=False)
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
            vocab_src.encode(sample["de"]), dtype=torch.int64, device=device
        )
        tgt_ids = torch.tensor(
            vocab_tgt.encode(sample["en"]), dtype=torch.int64, device=device
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

    train_data, valid_data, _ = load_multi30k()
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

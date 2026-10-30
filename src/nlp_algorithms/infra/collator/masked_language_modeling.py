from typing import List, Dict, Any

import random

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from nlp_algorithms.infra.collator import BaseCollator
from nlp_algorithms.engine.registry import register_collator


class MaskedSentencePairDataset(Dataset):
    """
    bert pretraining examples from documents of encoded sentences:
        <sos> A <eos> B <eos>, B is the text after A (is_next=1) or from another
        document (is_next=0), then 15% of the tokens are masked (80% <mask>,
        10% random token, 10% unchanged).
    the pairs and masks are sampled again every epoch, fixed=True seeds them by
    index so the validation loss is comparable between epochs.
    """

    def __init__(self, base_dataset, vocab, max_len, mask_prob=0.15, fixed=False):
        docs = [[s for s in ex["sentences"] if len(s)] for ex in base_dataset]
        docs = [d for d in docs if d]
        # one flat tensor + offsets, a tensor per sentence is ~270k storages on imdb and
        # the dataloader workers mmap each one (hits vm.max_map_count)
        sentences = [s for d in docs for s in d]
        self.tokens = torch.cat(sentences)
        self.sent_starts = torch.tensor([0] + [len(s) for s in sentences]).cumsum(0)
        self.doc_starts = torch.tensor([0] + [len(d) for d in docs]).cumsum(0)
        self.vocab = vocab
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.fixed = fixed
        self.mask_id = vocab.token_to_id["<mask>"]
        # pad, unk, sos, eos and mask are never masked or used as random tokens
        self.num_special = 5

    def __len__(self):
        return len(self.doc_starts) - 1

    def _num_sentences(self, doc):
        return int(self.doc_starts[doc + 1] - self.doc_starts[doc])

    def _span(self, doc, start, end):
        # tokens of sentences start..end-1 of a document
        first = int(self.doc_starts[doc])
        return self.tokens[self.sent_starts[first + start] : self.sent_starts[first + end]]

    def _sample_pair(self, idx, rng):
        n = self._num_sentences(idx)
        if n > 1:
            split = rng.randint(1, n - 1)
            a = self._span(idx, 0, split)
            if rng.random() < 0.5:
                return a, self._span(idx, split, n), 1
        else:
            a = self._span(idx, 0, 1)

        other = rng.randrange(len(self) - 1)
        other += other >= idx
        n_other = self._num_sentences(other)
        return a, self._span(other, rng.randrange(n_other), n_other), 0

    def _mask(self, input_ids, generator):
        labels = input_ids.clone()
        probs = torch.full(labels.shape, self.mask_prob)
        probs[input_ids < self.num_special] = 0.0
        masked = torch.bernoulli(probs, generator=generator).bool()
        labels[~masked] = -100

        r = torch.rand(labels.shape, generator=generator)
        input_ids = input_ids.clone()
        input_ids[masked & (r < 0.8)] = self.mask_id
        random_tokens = masked & (r >= 0.8) & (r < 0.9)
        input_ids[random_tokens] = torch.randint(
            self.num_special,
            len(self.vocab),
            (int(random_tokens.sum()),),
            generator=generator,
        )
        return input_ids, labels

    def __getitem__(self, idx):
        if self.fixed:
            rng, generator = random.Random(idx), torch.Generator().manual_seed(idx)
        else:
            rng, generator = random, None
        a, b, is_next = self._sample_pair(idx, rng)

        # keep the end of A and the start of B, they are next to each other
        budget = self.max_len - 3
        len_a = min(len(a), max(budget // 2, budget - len(b)))
        len_b = min(len(b), budget - len_a)
        a, b = a[len(a) - len_a :], b[:len_b]

        sos = torch.tensor([self.vocab.sos_id])
        eos = torch.tensor([self.vocab.eos_id])
        input_ids = torch.cat([sos, a, eos, b, eos])
        token_type_ids = torch.cat(
            [
                torch.zeros(len_a + 2, dtype=torch.long),
                torch.ones(len_b + 1, dtype=torch.long),
            ]
        )
        input_ids, mlm_labels = self._mask(input_ids, generator)
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "mlm_labels": mlm_labels,
            "is_next": is_next,
        }


@register_collator("masked_language_modeling")
class MaskedLanguageModelingCollator(BaseCollator):
    def __init__(self, vocab, architecture="bert"):
        super().__init__()
        self.vocab = vocab
        self.architecture = architecture

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch],
            batch_first=True,
            padding_value=self.vocab.pad_id,
        )
        token_type_ids = pad_sequence(
            [item["token_type_ids"] for item in batch], batch_first=True
        )
        mlm_labels = pad_sequence(
            [item["mlm_labels"] for item in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": (input_ids != self.vocab.pad_id).long(),
            "mlm_labels": mlm_labels,
            "nsp_labels": torch.tensor([item["is_next"] for item in batch]),
        }

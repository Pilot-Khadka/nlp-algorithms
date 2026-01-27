from typing import Optional, Tuple, Iterator

import torch

from engine.registry import register_collator


@register_collator("language_modeling")
class CorpusLanguageModelingDataset:
    def __init__(
        self,
        base_dataset,
        tokenizer,
        vocab,
        batch_size: int,
        seq_len: int,
        device: Optional[torch.device] = None,
        batch_first: bool = True,
    ):
        """
        Args:
            base_dataset: PTBDataset (contains single text item)
            tokenizer: Tokenizer instance
            vocab: Vocabulary instance
            batch_size: Number of parallel sequences
            seq_len: Length of each sequence chunk
            device: Device to place tensors on
            batch_first: If True, return (batch_size, seq_len), else (seq_len, batch_size)
        """
        self.base = base_dataset
        self.tokenizer = tokenizer()
        self.vocab = vocab
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self.batch_first = batch_first

        self._prepare_data()

    def _prepare_data(self):
        """Tokenize and encode the entire corpus, then batchify."""
        item = self.base[0]  # PTB has single item
        text = item["text"]

        tokens = self.tokenizer.tokenize(text)
        encoded = self.vocab.encode(tokens)
        data = torch.tensor(encoded, dtype=torch.long)

        print(f"Total tokens in corpus: {len(data)}")

        self.data = self._batchify(data, self.batch_size)
        print(f"Batchified shape: {self.data.shape}")  # (num_steps, batch_size)

        self.n_steps = self.data.size(0)
        self.n_batches = (self.n_steps - 1) // self.seq_len

    def _batchify(self, data: torch.Tensor, batch_size: int) -> torch.Tensor:
        """
        Reshape flat token sequence into parallel streams.

        Example with batch_size=3 and data=[0,1,2,3,4,5,6,7,8,9,10,11]:
            stream 0: [0, 1, 2, 3]
            stream 1: [4, 5, 6, 7]
            stream 2: [8, 9, 10, 11]

            result shape: (4, 3)
            [[0, 4, 8],
             [1, 5, 9],
             [2, 6, 10],
             [3, 7, 11]]
        """
        n_tokens = data.size(0)
        n_batches = n_tokens // batch_size

        data = data.narrow(0, 0, n_batches * batch_size)
        data = data.view(batch_size, -1).t().contiguous()

        return data

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate through corpus in seq_len chunks."""
        self.pos = 0
        return self

    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        outputs:
            x: (batch_size, seq_len) - input sequence
            y: (batch_size, seq_len) - target sequence (x shifted by 1)
        """
        if self.pos >= self.n_steps - 1:
            raise StopIteration

        actual_seq_len = min(self.seq_len, self.n_steps - 1 - self.pos)

        x = self.data[self.pos : self.pos + actual_seq_len]
        y = self.data[self.pos + 1 : self.pos + 1 + actual_seq_len]

        self.pos += actual_seq_len

        if self.device is not None:
            x = x.to(self.device)
            y = y.to(self.device)

        if self.batch_first:
            x = x.t().contiguous()  # (batch_size, seq_len)
            y = y.t().contiguous()

        return x, y

    def __len__(self) -> int:
        return self.n_batches

    def reset(self):
        self.pos = 0


import abc
import torch
from collections import Counter
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Any


class DatasetBundle:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        vocab=None,
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.vocab = vocab

    @property
    def vocab_size(self):
        assert self.vocab is not None
        return len(self.vocab)

    @property
    def num_classes(self):
        return self.train_loader.dataset.num_classes

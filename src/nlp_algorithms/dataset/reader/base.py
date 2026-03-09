from typing import Dict, List, Any
from abc import ABC, abstractmethod

from torch.utils.data import Dataset


class BaseTaskDataset(Dataset, ABC):
    def __init__(self, data_dir: str, split: str):
        self.data_dir = data_dir
        self.split = split
        self._validate_prepared()

    def _validate_prepared(self):
        import os

        prepared_flag = os.path.join(self.data_dir, ".prepared")
        if not os.path.exists(prepared_flag):
            raise RuntimeError(
                f"Data not prepared in {self.data_dir}. "
                f"Call the appropriate Downloader.download_and_prepare(cfg) first."
            )

    @abstractmethod
    def _load_raw_data(self) -> Any:
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class ClassificationDataset(BaseTaskDataset):
    """
    outputs:
    Returns format: {"text": str, "label": int, ...}
    :text: input text

    :label: integer label (0, 1, 2, ...)
    :additional fields allowed but these two are required
    """

    def __init__(self, data_dir: str, split: str):
        super().__init__(data_dir, split)
        self.examples: List[Dict[str, Any]] = []

    @abstractmethod
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load classification examples.

        Returns:
            List of dicts, each containing at minimum:
            - "text": str
            - "label": int
        """
        pass

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return a classification example."""
        example = self.examples[index]

        # Enforce required fields
        if "text" not in example or "label" not in example:
            raise ValueError(
                f"Classification dataset must return 'text' and 'label'. "
                f"Got keys: {example.keys()}"
            )

        # Validate types
        if not isinstance(example["text"], str):
            raise TypeError(f"'text' must be str, got {type(example['text'])}")
        if not isinstance(example["label"], int):
            raise TypeError(f"'label' must be int, got {type(example['label'])}")

        return example

    def __len__(self) -> int:
        return len(self.examples)


class LanguageModelingDataset(BaseTaskDataset):
    """Base class for language modeling tasks.

    Returns format: {"text": str}
    - text: raw text for language modeling
    """

    def __init__(self, data_dir: str, split: str):
        super().__init__(data_dir, split)
        self.text: str = ""

    @abstractmethod
    def _load_raw_data(self) -> str:
        """Load language modeling text.

        Returns:
            Single string containing all text for this split.
        """
        pass

    def __getitem__(self, index: int) -> Dict[str, str]:
        """Return the text sample."""
        if index != 0:
            raise IndexError(
                f"LanguageModelingDataset contains a single text sample. "
                f"Got index {index}"
            )

        return {"text": self.text}

    def __len__(self) -> int:
        return 1  # Single document per split


class Seq2SeqDataset(BaseTaskDataset):
    """Base class for sequence-to-sequence tasks (translation, summarization,
    etc).

    Returns format: {"src": str, "tgt": str}
    - src: source sequence
    - tgt: target sequence
    """

    def __init__(self, data_dir: str, split: str):
        super().__init__(data_dir, split)
        self.examples: List[Dict[str, str]] = []

    @abstractmethod
    def _load_raw_data(self) -> List[Dict[str, str]]:
        """Load sequence-to-sequence examples.

        Returns:
            List of dicts, each containing:
            - "src": str (source text)
            - "tgt": str (target text)
        """
        pass

    def __getitem__(self, index: int) -> Dict[str, str]:
        """Return a seq2seq example."""
        example = self.examples[index]

        # Enforce required fields
        if "src" not in example or "tgt" not in example:
            raise ValueError(
                f"Seq2SeqDataset must return 'src' and 'tgt'. "
                f"Got keys: {example.keys()}"
            )

        # Validate types
        if not isinstance(example["src"], str):
            raise TypeError(f"'src' must be str, got {type(example['src'])}")
        if not isinstance(example["tgt"], str):
            raise TypeError(f"'tgt' must be str, got {type(example['tgt'])}")

        return example

    def __len__(self) -> int:
        return len(self.examples)

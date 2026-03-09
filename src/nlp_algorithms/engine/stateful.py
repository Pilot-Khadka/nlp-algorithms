from abc import ABC, abstractmethod
from typing import Any, Optional


class StatefulModelMixin(ABC):
    """
    Mixin for models that maintain internal state across forward calls.
    Useful for RNNs, Transformers with KV cache, etc.
    """

    @property
    def is_stateful(self) -> bool:
        return True

    @abstractmethod
    def reset_state(self) -> None:
        """call at epoch/sequence start."""
        pass

    @abstractmethod
    def detach_state(self) -> None:
        """from computation graph for truncated BPTT."""
        pass

    def get_state(self) -> Optional[Any]:
        return None

    def set_state(self, state: Optional[Any]) -> None:
        pass

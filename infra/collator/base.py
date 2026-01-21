from typing import List, Dict, Any

from torch import Tensor


class BaseCollator:
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        return self.collate(batch)

    def collate(self, batch: List[Dict[str, Any]]) -> Dict[str, Tensor]:
        raise NotImplementedError

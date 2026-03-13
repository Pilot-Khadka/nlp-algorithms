import torch.nn as nn

from .base import BaseTask
from nlp_algorithms.engine.registry import register_task


@register_task("language_modeling")
class LMTask(BaseTask):
    allowed_flags = {"pytorch", "causal", "weight_tying", "use_locked_dropout"}

    @staticmethod
    def get_output_dim(dataset_bundle):
        return len(dataset_bundle.vocab)

    @staticmethod
    def get_loss(pad_idx=0):
        return nn.CrossEntropyLoss(ignore_index=pad_idx)

    def postprocess(self, outputs):
        raise NotImplementedError()

    def compute_metrics(self, outputs, batch):
        raise NotImplementedError()

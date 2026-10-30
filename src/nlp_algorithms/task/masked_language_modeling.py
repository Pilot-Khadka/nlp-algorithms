import torch.nn as nn

from .base import BaseTask
from nlp_algorithms.engine.registry import register_task


@register_task("masked_language_modeling")
class MaskedLanguageModelingTask(BaseTask):
    allowed_flags = set()

    @staticmethod
    def get_output_dim(dataset_bundle):
        return len(dataset_bundle.vocab)

    @staticmethod
    def get_loss():
        # unmasked positions have label -100, nsp labels are 0/1
        return nn.CrossEntropyLoss(ignore_index=-100)

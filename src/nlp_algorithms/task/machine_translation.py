from .base import BaseTask
from nlp_algorithms.engine.registry import register_task
from nlp_algorithms.encoder_decoder.loss import LabelSmoothing


@register_task("machine_translation")
class MachineTranslationTask(BaseTask):
    allowed_flags = {"shared_embeddings"}

    @staticmethod
    def get_output_dim(dataset_bundle):
        return len(dataset_bundle.tgt_vocab)

    @staticmethod
    def get_loss(size, pad_idx, smoothing=0.1):
        return LabelSmoothing(size=size, padding_idx=pad_idx, smoothing=smoothing)

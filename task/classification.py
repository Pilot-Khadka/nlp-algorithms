from .base import BaseTask

from engine.registry import register_task


@register_task("classification")
class ClassificationTask(BaseTask):
    allowed_flags = {"unidirectional", "bidirectional", "transformer"}

    @staticmethod
    def get_output_dim(dataset_bundle):
        return len(dataset_bundle.label_vocab)

    @staticmethod
    def get_loss():
        import torch.nn as nn

        return nn.CrossEntropyLoss()

    def postprocess(self, outputs):
        return outputs.argmax(dim=-1)

    def compute_metrics(self, outputs, batch):
        preds = outputs.argmax(dim=-1)
        labels = batch["labels"]
        acc = (preds == labels).float().mean().item()
        return {"accuracy": acc}

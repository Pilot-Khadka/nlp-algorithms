from abc import abstractmethod


class BaseTask:
    allowed_model_flags = set()  # e.g. {"causal", "unidirectional"}
    required_model_flags = set()  # if any

    @staticmethod
    @abstractmethod
    def get_output_dim(dataset_bundle) -> int:
        raise NotImplementedError

    def build_loss(self):
        raise NotImplementedError

    def compute_metrics(self, outputs, batch):
        return {}

    def postprocess(self, outputs):
        raise NotImplementedError

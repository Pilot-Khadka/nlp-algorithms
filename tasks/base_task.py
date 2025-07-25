from abc import ABC, abstractmethod


class BaseTask(ABC):
    @abstractmethod
    def get_output_dim(self, dataset_bundle):
        raise NotImplementedError()

    @abstractmethod
    def train_step(self, batch, model, criterion, optimizer, device):
        raise NotImplementedError

    @abstractmethod
    def eval_step(self, batch, model, criterion, device):
        raise NotImplementedError

    def compute_metrics(self, outputs, targets):
        return {}

from typing import Any, Union, Optional

import os
import pickle
import inspect
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from infra.preprocessor import PreprocessedDataset
from engine.registry import (
    TASK_REGISTRY,
    COLLATOR_REGISTRY,
    TOKENIZER_REGISTRY,
    get_from_registry,
)
from engine.model_factory import ModelFactory
from engine.optimizer import get_optimizer
from engine.dataset_builder import DatasetBundleBuilder, _is_trainable_tokenizer
from util.util import get_num_workers, resolve_tokenizer_path

ModelLike = Union[torch.nn.Module, DataParallel, DistributedDataParallel]
SchedulerLike = Optional[Union[_LRScheduler, ReduceLROnPlateau]]


@dataclass
class TrainerBundle:
    model: ModelLike
    optimizer: torch.optim.Optimizer
    scheduler: SchedulerLike
    train_loader: Any  # can be both Dataloader and iterator
    test_loader: Any  # can be both dataloader and iterator
    task: Any
    criterion: torch.nn.Module
    metric_names: list


class TrainerBuilder:
    def __init__(
        self,
        config,
        gpu_id: int = 0,
        use_ddp: bool = False,
    ):
        self.config = config
        self.gpu_id = gpu_id
        self.use_ddp = use_ddp
        self.device = torch.device(  # type: ignore[assignment]
            f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
        )

    def build(self) -> TrainerBundle:
        task = self._build_task()
        data_bundle = self._build_datasets()
        model = self._build_model(data_bundle)
        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        train_loader, test_loader = self._build_dataloaders(data_bundle)
        criterion = task.get_loss()
        metric_names = self._extract_metric_names()

        return TrainerBundle(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            task=task,
            criterion=criterion,
            metric_names=metric_names,
        )

    def _build_task(self):
        return get_from_registry(TASK_REGISTRY, self.config.task.name)

    def _build_datasets(self):
        return DatasetBundleBuilder().build(config=self.config)

    def _build_model(self, data_bundle):
        model = ModelFactory().create_model(
            model_config=self.config.model,
            dataset_bundle=data_bundle,
            task_config=self.config.task,
        )

        model.to(self.device)

        if self.config.get("train", {}).get("compile_model", False):
            print(f"Compiling model on GPU {self.gpu_id}...")
            model = torch.compile(model)

        if self.use_ddp:
            model = DDP(model, device_ids=[self.gpu_id])

        return model

    def _build_optimizer(self, model) -> torch.optim.Optimizer:
        return get_optimizer(model, train_config=self.config.train)

    def _build_scheduler(self, optimizer):
        if (
            hasattr(self.config, "optimizer")
            and self.config.optimizer.lower() == "sgd"
            and hasattr(self.config, "lr_decay")
            and self.config.lr_decay
        ):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.lr_decay,
                patience=1,
            )
        return None

    def _is_language_modeling_task(self) -> bool:
        return self.config.task.name == "language_modeling"

    def _build_dataloaders(self, data_bundle):
        if self._is_language_modeling_task():
            return self._build_language_modeling_loaders(data_bundle)

        return self.build_standard_dataloaders(data_bundle)

    def _build_language_modeling_loaders(self, data_bundle):
        tokenizer_cls = get_from_registry(
            TOKENIZER_REGISTRY, self.config.tokenizer.name
        )

        tokenizer_kwargs = {}
        sig = inspect.signature(tokenizer_cls.__init__)
        if "vocab_size" in sig.parameters:
            bpe_vocab_size = getattr(
                self.config.tokenizer, "vocab_size", self.config.dataset.vocab_size
            )
            tokenizer_kwargs["vocab_size"] = bpe_vocab_size

        tokenizer = tokenizer_cls(**tokenizer_kwargs)
        if _is_trainable_tokenizer(tokenizer):
            checkpoint_dir = getattr(self.config, "checkpoint_dir", "./checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            pickle_path = resolve_tokenizer_path(self.config)

            if os.path.exists(pickle_path):
                print(f"[LM] Loading tokenizer from: {pickle_path}")
                with open(pickle_path, "rb") as f:
                    state = pickle.load(f)
                tokenizer.vocab = state["vocab"]
                tokenizer.merges = state["merges"]

            else:
                raise RuntimeError(
                    f"[LM] Trainable tokenizer requires a saved .pkl file, "
                    f"but none found at: {pickle_path}\n"
                    f"Tokenizer must be trained earlier in DatasetBundleBuilder."
                )
        collator_class = get_from_registry(COLLATOR_REGISTRY, self.config.task.name)

        train_iterator = collator_class(
            base_dataset=data_bundle.train,
            tokenizer=tokenizer,
            vocab=data_bundle.vocab,
            batch_size=self.config.train.batch_size,
            seq_len=self.config.dataset.sequence_length,
            device=self.device,
            batch_first=True,
        )

        test_iterator = collator_class(
            base_dataset=data_bundle.test,
            tokenizer=tokenizer,
            vocab=data_bundle.vocab,
            batch_size=self.config.train.batch_size,
            seq_len=self.config.dataset.sequence_length,
            device=self.device,
            batch_first=True,
        )

        if self.use_ddp:
            print("Warning: DDP is not supported with language modeling iterators.")
            print(
                "Consider implementing custom distributed logic for language modeling."
            )

        return train_iterator, test_iterator

    def build_standard_dataloaders(self, data_bundle) -> tuple[DataLoader, DataLoader]:
        tokenizer_cls = get_from_registry(
            TOKENIZER_REGISTRY, self.config.tokenizer.name
        )

        tokenizer_kwargs = {}
        sig = inspect.signature(tokenizer_cls.__init__)
        if "vocab_size" in sig.parameters:
            bpe_vocab_size = getattr(
                self.config.tokenizer, "vocab_size", self.config.dataset.vocab_size
            )
            tokenizer_kwargs["vocab_size"] = bpe_vocab_size

        tokenizer = tokenizer_cls(**tokenizer_kwargs)

        if _is_trainable_tokenizer(tokenizer):
            checkpoint_dir = getattr(self.config, "checkpoint_dir", "./checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            pickle_path = os.path.join(checkpoint_dir, "lm_tokenizer.pkl")

            if os.path.exists(pickle_path):
                print(f"[LM] Loading tokenizer from: {pickle_path}")
                with open(pickle_path, "rb") as f:
                    state = pickle.load(f)
                    tokenizer.vocab = state["vocab"]
                    tokenizer.merges = state["merges"]

            else:
                raise RuntimeError(
                    f"[LM] Trainable tokenizer requires a saved .pkl file, "
                    f"but none found at: {pickle_path}\n"
                    f"Tokenizer must be trained earlier in DatasetBundleBuilder."
                )

        processed_train = PreprocessedDataset(
            data_bundle.train,
            tokenizer,
            data_bundle.vocab,
            max_len=self.config.dataset.sequence_length,
            task=self.config.task.name,
        )

        processed_test = PreprocessedDataset(
            data_bundle.test,
            tokenizer,
            data_bundle.vocab,
            max_len=self.config.dataset.sequence_length,
            task=self.config.task.name,
        )

        collator = get_from_registry(COLLATOR_REGISTRY, self.config.task.name)(
            vocab=data_bundle.vocab,
            architecture=self.config.model.name,
        )

        num_workers = self.config.dataset.get("num_workers", "auto")
        num_workers = get_num_workers(num_workers=num_workers)
        pin_memory = self.config.dataset.get("pin_memory", True)
        prefetch_factor = self.config.dataset.get("prefetch_factor", 2)

        if self.use_ddp:
            train_sampler = DistributedSampler(
                processed_train,
                shuffle=True,
            )
            test_sampler = DistributedSampler(
                processed_test,
                shuffle=False,
            )

            train_loader = DataLoader(
                processed_train,
                batch_size=self.config.train.batch_size,
                sampler=train_sampler,
                collate_fn=collator.collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )

            test_loader = DataLoader(
                processed_test,
                batch_size=self.config.train.batch_size,
                sampler=test_sampler,
                collate_fn=collator.collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )
        else:
            train_loader = DataLoader(
                processed_train,
                batch_size=self.config.train.batch_size,
                shuffle=True,
                collate_fn=collator.collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )

            test_loader = DataLoader(
                processed_test,
                batch_size=self.config.train.batch_size,
                shuffle=False,
                collate_fn=collator.collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
            )

        return train_loader, test_loader

    def _extract_metric_names(self) -> list:
        metrics = self.config.task.metrics
        if isinstance(metrics, dict):
            return list(metrics.keys())
        return metrics

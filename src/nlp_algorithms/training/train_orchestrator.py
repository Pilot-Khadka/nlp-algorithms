import os
import time
import logging

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from nlp_algorithms.util.logger import setup_logging, fmt_time
from nlp_algorithms.training.builder import TrainerBuilder
from nlp_algorithms.engine.registry import (
    DOWNLOADER_REGISTRY,
    TRAINER_REGISTRY,
    get_from_registry,
)


def ddp_setup(rank: int, world_size: int, master_port: str = "12355"):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def train_worker(rank: int, world_size: int, config):
    """
    description: worker function for distributed training on a single GPU.

    inputs:
        rank: GPU rank (0 to world_size-1)
        world_size: Total number of GPUs
        config: Training configuration
    """
    ddp_setup(rank, world_size)

    # rank 0 logs, the other ranks only warnings and errors
    logger = setup_logging()
    if rank != 0:
        logger.setLevel(logging.WARNING)

    try:
        pipeline = TrainerBuilder(
            config=config,
            gpu_id=rank,
            use_ddp=True,
        ).build()

        trainer_class = get_from_registry(TRAINER_REGISTRY, config.task.name)

        trainer = trainer_class(
            config=config,
            builder=pipeline,
            gpu_id=rank,
            use_ddp=True,
        )

        trainer.train()

    finally:
        cleanup()


def train_single_gpu(config, gpu_id: int = 0):
    """
    description: run training on a single GPU or CPU.

    inputs:
        config: Training configuration
        gpu_id: GPU device ID to use
    """
    pipeline = TrainerBuilder(
        config=config,
        gpu_id=gpu_id,
        use_ddp=False,
    ).build()

    trainer_class = get_from_registry(TRAINER_REGISTRY, config.task.name)

    trainer = trainer_class(
        config=config,
        builder=pipeline,
        gpu_id=gpu_id,
        use_ddp=False,
    )

    trainer.train()


def prepare_datasets_before_spawn(config):
    logger = setup_logging()
    start = time.perf_counter()

    dataset_class = get_from_registry(DOWNLOADER_REGISTRY, config.dataset.name)
    dataset_class().download_and_prepare(config)

    # train the tokenizer and write the preprocessed cache once, otherwise every
    # worker does it at the same time
    builder = TrainerBuilder(config=config, use_ddp=False)
    if not builder._is_language_modeling_task():
        builder.build_standard_dataloaders(builder._build_datasets())

    logger.info(f"data prep done in {fmt_time(time.perf_counter() - start)}")


def run_training(config):
    """
    description: main entry point for training.
        auto detects and configures multi-GPU or single-GPU training.

    inputs:
        config: Training configuration object
    """
    if config.task.name == "machine_translation":
        # the encoder-decoder has its own pipeline (token batches, ddp, beam search)
        from nlp_algorithms.training.machine_translation import run_machine_translation

        return run_machine_translation(config)

    logger = setup_logging()
    multi_gpu = config.train.get("multi_gpu", False)
    gpu_id = config.get("gpu_id", 0)

    if multi_gpu and torch.cuda.is_available():
        world_size = torch.cuda.device_count()

        if world_size > 1:
            logger.info(f"ddp on {world_size} gpus")
            prepare_datasets_before_spawn(config)
            mp.spawn(
                train_worker,
                args=(world_size, config),
                nprocs=world_size,
                join=True,
            )
        else:
            logger.warning("multi_gpu is set but there is only 1 gpu")
            train_single_gpu(config, gpu_id)
    else:
        if multi_gpu:
            logger.warning("multi_gpu is set but cuda is not available, training on cpu")
        train_single_gpu(config, gpu_id)

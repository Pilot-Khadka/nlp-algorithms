import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from util.logger import setup_logging
from training.builder import TrainerBuilder
from engine.registry import DOWNLOADER_REGISTRY, TRAINER_REGISTRY, get_from_registry


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

    logger = setup_logging() if rank == 0 else None

    try:
        if rank == 0 and logger:
            logger.info(f"Starting distributed training on {world_size} GPUs")
            logger.info(f"Worker {rank}: Building pipeline...")

        pipeline = TrainerBuilder(
            config=config,
            gpu_id=rank,
            use_ddp=True,
        ).build()

        if rank == 0 and logger:
            logger.info(f"Pipeline built successfully on rank {rank}")
            logger.info(f"Task metrics: {pipeline.metric_names}")

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
    logger = setup_logging()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Building training pipeline...")
    pipeline = TrainerBuilder(
        config=config,
        gpu_id=gpu_id,
        use_ddp=False,
    ).build()

    logger.info("Pipeline built successfully")
    logger.info(f"Model: {type(pipeline.model).__name__}")
    logger.info(f"Task metrics: {pipeline.metric_names}")

    trainer_class = get_from_registry(TRAINER_REGISTRY, config.task.name)

    trainer = trainer_class(
        config=config,
        builder=pipeline,
        gpu_id=gpu_id,
        use_ddp=False,
    )

    trainer.train()


def prepare_datasets_before_spawn(config):
    print("=" * 60)
    print("Preparing datasets in main process before spawning workers...")

    dataset_class = get_from_registry(DOWNLOADER_REGISTRY, config.dataset.name)
    dataset_class().download_and_prepare(config)

    print("Datasets ready!")
    print("=" * 60)


def run_training(config):
    """
    description: main entry point for training.
        auto detects and configures multi-GPU or single-GPU training.

    inputs:
        config: Training configuration object
    """
    multi_gpu = config.train.get("multi_gpu", False)
    gpu_id = config.get("gpu_id", 0)

    if multi_gpu and torch.cuda.is_available():
        world_size = torch.cuda.device_count()

        if world_size > 1:
            print(f"Multi-GPU training enabled: {world_size} GPUs detected")

            prepare_datasets_before_spawn(config)

            print(f"Spawning {world_size} training workers...")
            mp.spawn(
                train_worker,
                args=(world_size, config),
                nprocs=world_size,
                join=True,
            )
        else:
            print("Multi-GPU requested but only 1 GPU available")
            print("Falling back to single-GPU training")
            train_single_gpu(config, gpu_id)
    else:
        if multi_gpu:
            print("Multi-GPU training requested but CUDA not available")
            print("Running in CPU mode")
        else:
            print("Single-GPU/CPU mode")

        train_single_gpu(config, gpu_id)

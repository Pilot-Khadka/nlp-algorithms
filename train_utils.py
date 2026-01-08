import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


from datasets.loader import ensure_dataset_exists, load_dataset
from engine.trainer import Trainer
from utils.logger import setup_logging

from engine.task_factory import load_task
from engine.model_factory import ModelFactory
from engine.optimizer import get_optimizer


def ddp_setup(rank: int, world_size: int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def prepare_dataloader(
    original_loader: DataLoader,
    is_distributed: bool = False,
):
    if is_distributed:
        sampler = DistributedSampler(
            original_loader.dataset,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=True,
        )
        return DataLoader(
            original_loader.dataset,
            batch_size=original_loader.batch_size,
            sampler=sampler,
            num_workers=original_loader.num_workers,
            pin_memory=True,
            collate_fn=getattr(original_loader, "collate_fn", None),
        )
    return original_loader


def train_worker(
    rank: int,
    world_size: int,
    cfg,
):
    ddp_setup(rank, world_size)

    logger = setup_logging() if rank == 0 else None

    if rank == 0:
        print(f"Worker {rank}: Loading pre-downloaded dataset...")

    dataset_bundle = load_dataset(cfg)
    task = load_task(cfg.tasks.name)

    factory = ModelFactory()
    model = factory.create_model(cfg.models, dataset_bundle, task).to(rank)

    if cfg.get("train", {}).get("compile_model", False):
        print("Compiling the model ....")
        model = torch.compile(model)

    model = DDP(model, device_ids=[rank])

    optimizer = get_optimizer(model, cfg)
    train_loader = prepare_dataloader(dataset_bundle.train_loader, is_distributed=True)
    valid_loader = prepare_dataloader(dataset_bundle.valid_loader, is_distributed=True)

    metrics_to_use = cfg.tasks.get("metrics", [])

    if rank == 0 and logger:
        logger.info(f"Task metrics: {metrics_to_use}")

    training_config = cfg.train

    checkpoint = cfg.train.get("checkpoint", None)
    trainer = Trainer(
        model=model,
        task=task,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        config=training_config,
        metrics=metrics_to_use,
        logger=logger,
        gpu_id=rank,
        use_ddp=True,
        resume_from=checkpoint,
    )

    try:
        trainer.train()
    finally:
        cleanup()


def run_training(cfg_resolved):
    multi_gpu = cfg_resolved.train.get("multi_gpu", False)
    gpu_id = cfg_resolved.get("gpu_id", 0)

    if multi_gpu and torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        if world_size > 1:
            print(f"Starting multi-GPU training on {world_size} GPUs")

            print("=" * 60)
            print("Preparing dataset in main process before spawning workers...")
            ensure_dataset_exists(cfg_resolved)
            print("Dataset ready! Spawning workers...")
            print("=" * 60)

            mp.spawn(
                train_worker,
                args=(world_size, cfg_resolved),
                nprocs=world_size,
                join=True,
            )
            return
        else:
            print("Running in Single GPU / CPU mode")
            train_single_gpu(cfg_resolved, gpu_id)

    else:
        print("Running in Single GPU / CPU mode")
        train_single_gpu(cfg_resolved, gpu_id)


def train_single_gpu(cfg, gpu_id: int = 0):
    logger = setup_logging()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_bundle = load_dataset(cfg)
    task = load_task(cfg.tasks.name)

    factory = ModelFactory()
    model = factory.create_model(
        cfg.models,
        dataset_bundle,
        task,
    )

    if cfg.get("train", {}).get("compile_model", False):
        print("Compiling the model ....")
        model = torch.compile(model)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.train.learning_rate, weight_decay=1e-4
    )

    metrics_to_use = cfg.tasks.get("metrics", [])
    logger.info(f"Task metrics: {metrics_to_use}")

    train_loader = dataset_bundle.train_loader
    valid_loader = dataset_bundle.valid_loader

    training_config = cfg.train
    checkpoint = cfg.train.get("checkpoint", None)
    trainer = Trainer(
        model=model,
        task=task,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        config=training_config,
        metrics=metrics_to_use,
        logger=logger,
        gpu_id=gpu_id,
        use_ddp=False,
        resume_from=checkpoint,
    )

    trainer.train()

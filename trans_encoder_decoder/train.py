import os
import json
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.distributed as dist
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from trans_encoder_decoder import Seq2Seq
from util.util import convert_to_attrdict, get_num_workers
from engine.dataset_builder import build_vocab_from_key
from infra.preprocessor import PreprocessedDataset
from engine.registry import (
    DATA_READER_REGISTRY,
    DOWNLOADER_REGISTRY,
    COLLATOR_REGISTRY,
    TOKENIZER_REGISTRY,
    get_from_registry,
)
from dataset.downloader import TatoebaDownloader


class DatasetBundle:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        train_sampler,
        test_sampler,
        label_vocab,
        src_vocab,
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.label_vocab = label_vocab
        self.src_vocab = src_vocab
        self.train_sampler = train_sampler
        self.test_sampler = test_sampler


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def create_masks(src, tgt, pad_idx=0):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_padding_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.size(1)
    tgt_causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device)).bool()
    tgt_mask = tgt_padding_mask & tgt_causal_mask
    return src_mask, tgt_mask


def train_epoch(
    model, dataloader, optimizer, criterion, device, pad_idx=0, rank=0, world_size=1
):
    model.train()
    total_loss = 0
    num_batches = 0

    if is_main_process(rank):
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    for batch in progress_bar:
        src_ids = batch["src_ids"].to(device)  # Already padded
        tgt_ids = batch["tgt_ids"].to(device)  # Already shifted (input)
        labels = batch["labels"].to(device)  # Already shifted (output)

        src_mask, tgt_mask = create_masks(src_ids, tgt_ids, pad_idx)

        optimizer.zero_grad()
        logits = model(src_ids, tgt_ids, src_mask, tgt_mask)

        if (labels >= logits.size(-1)).any() or (labels < 0).any():
            print(
                "BAD LABEL:",
                labels.min().item(),
                labels.max().item(),
                "num_classes:",
                logits.size(-1),
            )
            raise ValueError("Label out of range")

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if is_main_process(rank):
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

    return avg_loss


def evaluate(model, dataloader, criterion, device, pad_idx=0, rank=0, world_size=1):
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        if is_main_process(rank):
            progress_bar = tqdm(dataloader, desc="Evaluating")
        else:
            progress_bar = dataloader

        for batch in progress_bar:
            # Get source and target from collator output
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            labels = batch["labels"].to(device)

            # Create masks
            src_mask, tgt_mask = create_masks(src_ids, tgt_ids, pad_idx)

            # Forward pass
            logits = model(src_ids, tgt_ids, src_mask, tgt_mask)

            # Compute loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    if world_size > 1:
        avg_loss_tensor = torch.tensor(avg_loss, device=device)
        dist.all_reduce(avg_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss_tensor.item() / world_size

    return avg_loss


def get_dataloaders_distributed(config, world_size):
    data_downloader_cls = get_from_registry(DOWNLOADER_REGISTRY, config.dataset.name)
    data_dir = data_downloader_cls().download_and_prepare(config)

    data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name)
    train = data_reader_cls(
        data_dir=data_dir,
        split="train",
        max_samples=config.dataset.max_samples,
    )
    test = data_reader_cls(
        data_dir=data_dir,
        split="test",
        max_samples=config.dataset.max_samples,
    )

    tokenizer = get_from_registry(TOKENIZER_REGISTRY, config.tokenizer.name)
    src_vocab = build_vocab_from_key(dataset=train, tokenizer=tokenizer, key="src")
    tgt_vocab = build_vocab_from_key(
        dataset=train, tokenizer=tokenizer, key="tgt"
    )  # note train for target also

    processed_train = PreprocessedDataset(
        train,
        tokenizer,
        vocab=src_vocab,
        target_vocab=tgt_vocab,
        max_len=config.dataset.sequence_length,
        task=config.task.name,
    )
    processed_test = PreprocessedDataset(
        test,
        tokenizer,
        vocab=src_vocab,
        target_vocab=tgt_vocab,
        max_len=config.dataset.sequence_length,
        task=config.task.name,
    )
    label_vocab = tgt_vocab

    collator = get_from_registry(COLLATOR_REGISTRY, config.task.name)()
    num_workers = config.dataset.get("num_workers", "auto")
    num_workers = get_num_workers(num_workers=num_workers)
    pin_memory = config.dataset.get("pin_memory", True)
    prefetch_factor = config.dataset.get("prefetch_factor", 2)

    if world_size > 1:
        train_sampler = DistributedSampler(processed_train, shuffle=True)
        test_sampler = DistributedSampler(processed_test, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = DataLoader(
        processed_train,
        batch_size=config.train.batch_size,
        sampler=train_sampler,
        collate_fn=collator.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    test_loader = DataLoader(
        processed_test,
        batch_size=config.train.batch_size,
        sampler=test_sampler,
        collate_fn=collator.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    bundle = DatasetBundle(
        train_loader=train_loader,
        valid_loader=test_loader,
        test_loader=test_loader,
        train_sampler=train_sampler,
        test_sampler=test_sampler,
        src_vocab=src_vocab,
        label_vocab=label_vocab,
    )
    return bundle


def train(cfg, rank, world_size, local_rank):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main_process(rank):
        print(f"Using device: {device}")
        print(f"World size: {world_size}")

    if is_main_process(rank):
        print("Loading dataset...")

    dataset_bundle = get_dataloaders_distributed(cfg, world_size=world_size)
    train_loader = dataset_bundle.train_loader
    val_loader = dataset_bundle.valid_loader
    test_loader = dataset_bundle.test_loader

    source_vocab_size = len(dataset_bundle.src_vocab)
    target_vocab_size = len(dataset_bundle.label_vocab)

    if is_main_process(rank):
        print(f"Source vocab size: {source_vocab_size}")
        print(f"Target vocab size: {target_vocab_size}")

    model = Seq2Seq(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
    ).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    if is_main_process(rank):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    if is_main_process(rank):
        os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)

    best_val_loss = float("inf")
    train_history = []

    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        if is_main_process(rank):
            print(f"\nEpoch {epoch}/{cfg['train']['num_epochs']}")

        if world_size > 1:
            dataset_bundle.train_sampler.set_epoch(epoch)

        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            rank=rank,
            world_size=world_size,
        )
        val_loss = evaluate(
            model, val_loader, criterion, device, rank=rank, world_size=world_size
        )

        scheduler.step(val_loss)

        if is_main_process(rank):
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")

            train_history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

                model_state_dict = (
                    model.module.state_dict()
                    if isinstance(model, DDP)
                    else model.state_dict()
                )

                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": cfg,
                }
                checkpoint_path = os.path.join(
                    cfg["train"]["checkpoint_dir"], "best_model.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best model (val_loss: {val_loss:.4f})")

            if epoch % cfg["train"]["save_every"] == 0:
                model_state_dict = (
                    model.module.state_dict()
                    if isinstance(model, DDP)
                    else model.state_dict()
                )

                checkpoint_path = os.path.join(
                    cfg["train"]["checkpoint_dir"], f"checkpoint_epoch_{epoch}.pt"
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model_state_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    },
                    checkpoint_path,
                )

        if world_size > 1:
            dist.barrier()

    if is_main_process(rank):
        history_path = os.path.join(
            cfg["train"]["checkpoint_dir"], "train_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(train_history, f, indent=2)

        print("\nEvaluating on test set...")

    test_loss = evaluate(
        model, test_loader, criterion, device, rank=rank, world_size=world_size
    )

    if is_main_process(rank):
        print(f"Test Loss: {test_loss:.4f}")

    return model, train_history


def main():
    cfg = {
        "dataset": {
            "name": "tatoeba",
            "url": "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eng-nep.tar",
            "data_dir": "../dataset/dataset_tatoeba_eng_nep/",
            "vocab_size": 10000,
            "sequence_length": 256,
            "max_samples": 1000000,
        },
        "model": {
            "d_model": 512,
            "num_layers": 6,
            "num_heads": 8,
            "d_ff": 2048,
        },
        "train": {
            "batch_size": 128,
            "num_epochs": 20,
            "learning_rate": 1e-4,
            "checkpoint_dir": "checkpoints",
            "save_every": 5,
        },
        "tokenizer": {"name": "whitespace"},
        "task": {"name": "translation"},
    }
    cfg = convert_to_attrdict(cfg)
    print("\n=== Downloading Dataset ===")
    TatoebaDownloader.download_and_prepare(cfg)
    print("Dataset ready!\n")

    rank, world_size, local_rank = setup_distributed()

    if is_main_process(rank):
        print("Configuration:")
        print(json.dumps(cfg, indent=2))

    model, history = train(cfg, rank, world_size, local_rank)

    if is_main_process(rank):
        print("\nTraining completed!")

    cleanup_distributed()


if __name__ == "__main__":
    main()

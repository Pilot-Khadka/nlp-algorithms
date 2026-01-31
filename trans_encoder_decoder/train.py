import os
import json
import math
from tqdm import tqdm
from collections import Counter
from typing import List


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


def calculate_bleu(
    references: List[List[str]], hypotheses: List[List[str]], max_n: int = 4
) -> dict:
    """
    Compute BLEU score for machine translation evaluation.

    Args:
        references: List of reference translations (each as list of tokens)
        hypotheses: List of hypothesis translations (each as list of tokens)
        max_n: Maximum n-gram order (default: 4)

    Returns:
        Dictionary with BLEU scores
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references and hypotheses must match")

    precisions = []

    for n in range(1, max_n + 1):
        ref_counts = Counter()
        hyp_counts = Counter()

        for ref, hyp in zip(references, hypotheses):
            ref_ngrams = [tuple(ref[i : i + n]) for i in range(len(ref) - n + 1)]
            hyp_ngrams = [tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1)]

            for ngram in ref_ngrams:
                ref_counts[ngram] += 1
            for ngram in hyp_ngrams:
                hyp_counts[ngram] += 1

        clipped_counts = sum(
            min(hyp_counts[ngram], ref_counts[ngram]) for ngram in hyp_counts
        )
        total_hyp_ngrams = sum(hyp_counts.values())

        if total_hyp_ngrams > 0:
            precision = clipped_counts / total_hyp_ngrams
        else:
            precision = 0.0

        precisions.append(precision)

    ref_length = sum(len(ref) for ref in references)
    hyp_length = sum(len(hyp) for hyp in hypotheses)

    if hyp_length > ref_length:
        bp = 1.0
    elif hyp_length == 0:
        bp = 0.0
    else:
        bp = math.exp(1 - ref_length / hyp_length)

    if min(precisions) > 0:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_n)
        bleu = bp * geo_mean
    else:
        bleu = 0.0

    return {
        "bleu": bleu * 100,
        "bleu-1": precisions[0] * 100,
        "bleu-2": precisions[1] * 100 if len(precisions) > 1 else 0.0,
        "bleu-3": precisions[2] * 100 if len(precisions) > 2 else 0.0,
        "bleu-4": precisions[3] * 100 if len(precisions) > 3 else 0.0,
        "brevity_penalty": bp,
        "ref_length": ref_length,
        "hyp_length": hyp_length,
    }


def greedy_decode(
    model, src, src_mask, max_len, tgt_vocab, device, pad_idx=0, sos_idx=1, eos_idx=2
):
    """
    Greedy decoding for translation.

    Args:
        model: The sequence-to-sequence model
        src: Source sequence [batch_size, src_len]
        src_mask: Source mask
        max_len: Maximum length for generation
        tgt_vocab: Target vocabulary
        device: Device to use
        pad_idx: Padding token index
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index

    Returns:
        Generated sequences [batch_size, max_len]
    """
    batch_size = src.size(0)

    generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = (
            torch.tril(torch.ones(generated.size(1), generated.size(1), device=device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            logits = model(src, generated, src_mask, tgt_mask)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if (next_token == eos_idx).all():
            break

    return generated


def tokens_to_words(token_ids, vocab, pad_idx=0, sos_idx=1, eos_idx=2):
    """
    Convert token IDs to words.

    Args:
        token_ids: Tensor or list of token IDs
        vocab: Vocabulary object
        pad_idx: Padding token index
        sos_idx: Start-of-sequence token index
        eos_idx: End-of-sequence token index

    Returns:
        List of words
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    words = []
    for token_id in token_ids:
        if token_id in [pad_idx, sos_idx, eos_idx]:
            continue
        word = vocab.lookup_token(token_id)
        words.append(word)

    return words


def train_epoch(
    model, dataloader, optimizer, criterion, device, pad_idx=0, rank=0, world_size=1
):
    """Train for one epoch and return average loss."""
    model.train()
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    if is_main_process(rank):
        progress_bar = tqdm(dataloader, desc="Training")
    else:
        progress_bar = dataloader

    for batch in progress_bar:
        src_ids = batch["src_ids"].to(device)
        tgt_ids = batch["tgt_ids"].to(device)
        labels = batch["labels"].to(device)

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

        non_pad_tokens = (labels != pad_idx).sum().item()
        batch_loss = loss.item() * logits.size(0) * logits.size(1)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss
        total_tokens += non_pad_tokens
        num_batches += 1

        if is_main_process(rank):
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_tokens_tensor = torch.tensor(total_tokens, device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)

        avg_loss = total_loss_tensor.item() / total_tokens_tensor.item()

    return avg_loss


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    tgt_vocab,
    compute_metrics=True,
    max_decode_len=80,
    pad_idx=0,
    rank=0,
    world_size=1,
):
    """
    inputs:
        model: The model to evaluate
        dataloader: DataLoader for evaluation
        criterion: Loss criterion
        device: Device to use
        tgt_vocab: Target vocabulary for decoding
        compute_metrics: Whether to compute BLEU scores
        max_decode_len: Maximum length for decoding
        pad_idx: Padding token index
        rank: Process rank
        world_size: Number of processes

    outputs:
        Dictionary with loss and metrics
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    all_references = []
    all_hypotheses = []

    with torch.no_grad():
        if is_main_process(rank):
            progress_bar = tqdm(dataloader, desc="Evaluating")
        else:
            progress_bar = dataloader

        for batch in progress_bar:
            src_ids = batch["src_ids"].to(device)
            tgt_ids = batch["tgt_ids"].to(device)
            labels = batch["labels"].to(device)

            src_mask, tgt_mask = create_masks(src_ids, tgt_ids, pad_idx)

            logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            non_pad_tokens = (labels != pad_idx).sum().item()
            batch_loss = loss.item() * logits.size(0) * logits.size(1)

            total_loss += batch_loss
            total_tokens += non_pad_tokens

            if compute_metrics:
                generated = greedy_decode(
                    model,
                    src_ids,
                    src_mask,
                    max_decode_len,
                    tgt_vocab,
                    device,
                    pad_idx=pad_idx,
                )

                for i in range(src_ids.size(0)):
                    ref_tokens = tokens_to_words(labels[i], tgt_vocab, pad_idx=pad_idx)
                    hyp_tokens = tokens_to_words(
                        generated[i], tgt_vocab, pad_idx=pad_idx
                    )

                    all_references.append(ref_tokens)
                    all_hypotheses.append(hyp_tokens)

    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0

    if world_size > 1:
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_tokens_tensor = torch.tensor(total_tokens, device=device)

        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens_tensor, op=dist.ReduceOp.SUM)

        avg_loss = total_loss_tensor.item() / total_tokens_tensor.item()

    results = {"loss": avg_loss}

    if compute_metrics and len(all_references) > 0:
        if world_size > 1:
            if is_main_process(rank):
                bleu_scores = calculate_bleu(all_references, all_hypotheses)
                results.update(bleu_scores)
        else:
            bleu_scores = calculate_bleu(all_references, all_hypotheses)
            results.update(bleu_scores)

    return results


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
    src_vocab = build_vocab_from_key(
        dataset=train,
        config=config,
        tokenizer=tokenizer,
        key="src",
    )
    tgt_vocab = build_vocab_from_key(
        dataset=train,
        config=config,
        tokenizer=tokenizer,
        key="tgt",
    )

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

    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")
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
    best_bleu = 0.0
    train_history = []

    for epoch in range(1, cfg["train"]["num_epochs"] + 1):
        if is_main_process(rank):
            print(f"\n{'=' * 60}")
            print(f"Epoch {epoch}/{cfg['train']['num_epochs']}")
            print(f"{'=' * 60}")

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

        val_results = evaluate(
            model,
            val_loader,
            criterion,
            device,
            dataset_bundle.label_vocab,
            compute_metrics=True,
            max_decode_len=cfg.dataset.sequence_length,
            rank=rank,
            world_size=world_size,
        )

        scheduler.step(val_results["loss"])

        if is_main_process(rank):
            print(f"\nTrain Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_results['loss']:.4f}")

            if "bleu" in val_results:
                print(f"Val BLEU: {val_results['bleu']:.2f}")
                print(f"  BLEU-1: {val_results['bleu-1']:.2f}")
                print(f"  BLEU-2: {val_results['bleu-2']:.2f}")
                print(f"  BLEU-3: {val_results['bleu-3']:.2f}")
                print(f"  BLEU-4: {val_results['bleu-4']:.2f}")
                print(f"  Brevity Penalty: {val_results['brevity_penalty']:.4f}")

            history_entry = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_results["loss"],
                "lr": optimizer.param_groups[0]["lr"],
            }

            if "bleu" in val_results:
                history_entry.update(
                    {
                        "val_bleu": val_results["bleu"],
                        "val_bleu_1": val_results["bleu-1"],
                        "val_bleu_2": val_results["bleu-2"],
                        "val_bleu_3": val_results["bleu-3"],
                        "val_bleu_4": val_results["bleu-4"],
                    }
                )

            train_history.append(history_entry)

            save_best = False
            save_reason = None
            if "bleu" in val_results and val_results["bleu"] > best_bleu:
                best_bleu = val_results["bleu"]
                save_best = True
                save_reason = f"BLEU: {best_bleu:.2f}"
            elif val_results["loss"] < best_val_loss:
                best_val_loss = val_results["loss"]
                save_best = True
                save_reason = f"Loss: {best_val_loss:.4f}"

            if save_best:
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
                    "val_results": val_results,
                    "config": cfg,
                }
                checkpoint_path = os.path.join(
                    cfg["train"]["checkpoint_dir"], "best_model.pt"
                )
                torch.save(checkpoint, checkpoint_path)
                print(f"Saved best model ({save_reason})")

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
                        "val_results": val_results,
                    },
                    checkpoint_path,
                )
                print(f"✓ Saved checkpoint at epoch {epoch}")

        if world_size > 1:
            dist.barrier()

    if is_main_process(rank):
        history_path = os.path.join(
            cfg["train"]["checkpoint_dir"], "train_history.json"
        )
        with open(history_path, "w") as f:
            json.dump(train_history, f, indent=2)

        print(f"\n{'=' * 60}")
        print("Evaluating on test set...")
        print(f"{'=' * 60}")

    test_results = evaluate(
        model,
        test_loader,
        criterion,
        device,
        dataset_bundle.label_vocab,
        compute_metrics=True,
        max_decode_len=cfg.dataset.sequence_length,
        rank=rank,
        world_size=world_size,
    )

    if is_main_process(rank):
        print(f"\n{'=' * 60}")
        print("Test Results:")
        print(f"{'=' * 60}")
        print(f"Test Loss: {test_results['loss']:.4f}")

        if "bleu" in test_results:
            print(f"Test BLEU: {test_results['bleu']:.2f}")
            print(f"  BLEU-1: {test_results['bleu-1']:.2f}")
            print(f"  BLEU-2: {test_results['bleu-2']:.2f}")
            print(f"  BLEU-3: {test_results['bleu-3']:.2f}")
            print(f"  BLEU-4: {test_results['bleu-4']:.2f}")
            print(f"  Brevity Penalty: {test_results['brevity_penalty']:.4f}")
            print(f"  Reference Length: {test_results['ref_length']}")
            print(f"  Hypothesis Length: {test_results['hyp_length']}")

        test_results_path = os.path.join(
            cfg["train"]["checkpoint_dir"], "test_results.json"
        )
        with open(test_results_path, "w") as f:
            json.dump(test_results, f, indent=2)

    return model, train_history, test_results


def main():
    cfg = {
        "dataset": {
            "name": "tatoeba",
            "url": "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eng-nep.tar",
            "data_dir": "../dataset/dataset_tatoeba_eng_nep/",
            "vocab_size": 10000,
            "sequence_length": 80,
            "max_samples": 1000000,
        },
        "model": {
            "d_model": 256,
            "num_layers": 3,
            "num_heads": 4,
            "d_ff": 1024,
        },
        "train": {
            "batch_size": 128,
            "num_epochs": 20,
            "learning_rate": 1e-4,
            "checkpoint_dir": "checkpoints",
            "save_every": 5,
        },
        "tokenizer": {"name": "bpe"},
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

    model, history, test_results = train(cfg, rank, world_size, local_rank)

    if is_main_process(rank):
        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)

        if "bleu" in test_results:
            print(f"Final Test BLEU: {test_results['bleu']:.2f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

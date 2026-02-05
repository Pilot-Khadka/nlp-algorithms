import os
import json
import inspect
from tqdm import tqdm

import sacrebleu

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from trans_attention.attention import (
    convert_to_additive,
    make_causal_mask,
    make_padding_mask,
)
from trans_encoder_decoder import Seq2Seq
from util.util import get_num_workers
from engine.dataset_builder import build_vocab_from_key
from infra.preprocessor import PreprocessedDataset
from engine.registry import (
    DATA_READER_REGISTRY,
    DOWNLOADER_REGISTRY,
    COLLATOR_REGISTRY,
    TOKENIZER_REGISTRY,
    get_from_registry,
)


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
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

    return rank, world_size, local_rank


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def create_masks(src, tgt, pad_idx):
    """
    description:
        encoder self-attention: padding mask only (src)
        decoder self-attention: causal+padding mask (tgt)
        cross-attention: padding mask for encoder source tokens (src)

    outputs:
        src padding mask, tgt combined mask

    """
    B, T = tgt.shape
    src_pad = make_padding_mask(seq=src, pad_idx=pad_idx).to(tgt.device)
    tgt_pad = make_padding_mask(seq=tgt, pad_idx=pad_idx).to(tgt.device)
    causal_mask = make_causal_mask(T).to(tgt.device)  # True = keep

    src_padding_mask = convert_to_additive(src_pad)
    tgt_combined_mask = convert_to_additive(tgt_pad & causal_mask)
    return src_padding_mask, tgt_combined_mask


def calculate_bleu(references, hypotheses):
    bleu = sacrebleu.corpus_bleu(
        hypotheses=hypotheses,
        references=[references],
    )

    return {
        "bleu": bleu.score,
        "bleu-1": bleu.precisions[0],
        "bleu-2": bleu.precisions[1],
        "bleu-3": bleu.precisions[2],
        "bleu-4": bleu.precisions[3],
        "brevity_penalty": bleu.bp,
        "hyp_length": bleu.sys_len,
        "ref_length": bleu.ref_len,
    }


def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    device,
    sos_idx,
    eos_idx,
):
    batch_size = src.size(0)
    generated = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)

    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        tgt_len = generated.size(1)

        tgt_mask = (
            torch.tril(torch.ones(tgt_len, tgt_len, device=device))
            .bool()
            .unsqueeze(0)
            .unsqueeze(0)
        )

        with torch.no_grad():
            logits = model(src, generated, src_mask, tgt_mask)

        next_token = logits[:, -1, :].argmax(dim=-1)

        next_token = torch.where(
            finished, torch.tensor(eos_idx, device=device), next_token
        )

        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | (next_token == eos_idx)

        if finished.all():
            break

    return generated


def tokens_to_words(
    token_ids,
    vocab,
    pad_idx=0,
    sos_idx=2,
    eos_idx=3,
    end_token="</w>",
):
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    if eos_idx in token_ids:
        token_ids = token_ids[: token_ids.index(eos_idx)]

    # remove pad + sos
    token_ids = [t for t in token_ids if t not in {pad_idx, sos_idx}]

    # Decode ids -> string tokens
    tokens = vocab.decode(token_ids)

    if isinstance(tokens, list):
        tokens = "".join(tokens)

    tokens = tokens.replace(end_token, " ")
    return tokens.strip()


def train_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    pad_idx=0,
    rank=0,
    world_size=1,
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

        if torch.isnan(logits).any():
            print("NAN in logits!")
            print("\n===== NAN DETECTED =====")
            print("src_ids:", src_ids)
            print("tgt_ids:", tgt_ids)
            print("labels:", labels)
            print(
                "logits stats:",
                torch.isnan(logits).any().item(),
                logits.min().item(),
                logits.max().item(),
            )
            print("tgt_ids unique:", torch.unique(tgt_ids))
            print("labels unique:", torch.unique(labels))
            print("Batch index:", num_batches)

            for i in range(src_ids.size(0)):
                print("---- SAMPLE", i, "----")
                print("src:", src_ids[i])
                print("tgt:", tgt_ids[i])
                print("label:", labels[i])
            raise RuntimeError()

        if torch.isinf(logits).any():
            print("INF in logits!")
            raise RuntimeError()

        if logits.abs().max() > 50:
            print("WARNING: Exploding logits:", logits.abs().max().item())

        loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

        non_pad_tokens = (labels != pad_idx).sum().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * non_pad_tokens
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


def clean_ids(ids, pad_idx, sos_idx, eos_idx):
    out = []
    for t in ids:
        if t == eos_idx:
            break
        if t not in {pad_idx, sos_idx}:
            out.append(t)
    return out


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
    sos_idx=2,
    eos_idx=3,
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

            # for i in range(src_ids.size(0)):
            #     print("target:", tokens_to_words(token_ids=tgt_ids[i], vocab=tgt_vocab))
            #     print("labels:", tokens_to_words(token_ids=labels[i], vocab=tgt_vocab))

            src_mask, tgt_mask = create_masks(src_ids, tgt_ids, pad_idx)

            logits = model(src_ids, tgt_ids, src_mask, tgt_mask)
            loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

            non_pad_tokens = (labels != pad_idx).sum().item()

            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens

            if compute_metrics:
                generated = greedy_decode(
                    model=model,
                    src=src_ids,
                    src_mask=src_mask,
                    max_len=max_decode_len,
                    device=device,
                    sos_idx=sos_idx,
                    eos_idx=eos_idx,
                )

                for i in range(src_ids.size(0)):
                    hyp_string = tokens_to_words(
                        token_ids=generated[i], vocab=tgt_vocab
                    )
                    ref_string = tokens_to_words(token_ids=labels[i], vocab=tgt_vocab)

                    all_references.append(ref_string)
                    all_hypotheses.append(hyp_string)

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


def init_tokenizer(tokenizer_name, config):
    tokenizer_cls = get_from_registry(TOKENIZER_REGISTRY, tokenizer_name)
    tokenizer_kwargs = {}
    sig = inspect.signature(tokenizer_cls.__init__)

    if "vocab_size" in sig.parameters:
        tokenizer_kwargs["vocab_size"] = getattr(
            config.tokenizer, "vocab_size", config.dataset.vocab_size
        )

    return tokenizer_cls(**tokenizer_kwargs)


def get_dataloaders_distributed(config, world_size):
    data_downloader_cls = get_from_registry(DOWNLOADER_REGISTRY, config.dataset.name)
    data_dir = data_downloader_cls().download_and_prepare(config)

    if config.dataset.name == "huggingface":
        data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name2)
    else:
        data_reader_cls = get_from_registry(DATA_READER_REGISTRY, config.dataset.name)

    max_samples = config.dataset.get("max_samples", None)
    train = data_reader_cls(
        data_dir=data_dir,
        split="train",
        max_samples=max_samples,
    )
    test = data_reader_cls(
        data_dir=data_dir,
        split="test",
        max_samples=max_samples,
    )

    tokenizer_src = init_tokenizer(config.tokenizer.name, config)
    tokenizer_tgt = init_tokenizer(config.tokenizer.name, config)

    src_vocab = build_vocab_from_key(
        dataset=train,
        config=config,
        tokenizer=tokenizer_src,
        key="src",
    )
    tgt_vocab = build_vocab_from_key(
        dataset=train,
        config=config,
        tokenizer=tokenizer_tgt,
        key="tgt",
    )

    processed_train = PreprocessedDataset(
        train,
        src_tokenizer=tokenizer_src,
        tgt_tokenizer=tokenizer_tgt,
        vocab=src_vocab,
        target_vocab=tgt_vocab,
        max_len=config.dataset.sequence_length,
        task=config.task.name,
    )
    processed_test = PreprocessedDataset(
        test,
        src_tokenizer=tokenizer_src,
        tgt_tokenizer=tokenizer_tgt,
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

        # scheduler.step(val_results["loss"])

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
                print(f" Saved checkpoint at epoch {epoch}")

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

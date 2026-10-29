import random
import numpy as np
from tqdm import tqdm
from nlp_algorithms.encoder_decoder.data import (
    load_ne_en,
    load_vocab,
    collate_batch,
    HFDatasetWrapper,
    DataLoader,
    DistributedSampler,
)
from torch.utils.data import Subset


def diagnose_tokenization(loader, vocab_src, vocab_tgt, pad_id, n=3):
    src_batch, tgt_batch = next(iter(loader))
    for i in range(min(n, src_batch.shape[0])):
        src_ids = src_batch[i]
        tgt_ids = tgt_batch[i]

        src_tokens = src_ids[src_ids != pad_id].tolist()
        tgt_tokens = tgt_ids[tgt_ids != pad_id].tolist()

        src_text = vocab_src.decode(src_tokens)
        tgt_text = vocab_tgt.decode(tgt_tokens)

        print(f"\n[{i}] src text  : {src_text}")
        print(f"     src token len (no pad): {len(src_tokens)} -> {src_tokens[:10]}...")
        print(f"     tgt text  : {tgt_text}")
        print(f"     tgt token len (no pad): {len(tgt_tokens)} -> {tgt_tokens[:10]}...")


def compute_length_stats(loader, pad_id, sample_size=10_000, desc=""):
    src_lens, tgt_lens = [], []

    for src_batch, tgt_batch in tqdm(loader, desc=desc):
        if len(src_lens) >= sample_size:
            break
        for src_row, tgt_row in zip(src_batch, tgt_batch):
            src_lens.append((src_row != pad_id).sum().item())
            tgt_lens.append((tgt_row != pad_id).sum().item())

    src_lens = src_lens[:sample_size]
    tgt_lens = tgt_lens[:sample_size]

    def stats(arr):
        arr = np.array(arr)
        return {
            "min": int(arr.min()),
            "mean": float(arr.mean()),
            "median": int(np.median(arr)),
            "max": int(arr.max()),
            "95p": int(np.percentile(arr, 95)),
        }

    return stats(src_lens), stats(tgt_lens)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    load_fn,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
    sample_size=10_000,
):
    def collate_fn(batch):
        return collate_batch(
            batch,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src["<blank>"],
        )

    train_data, valid_data, _ = load_fn()
    train_data = HFDatasetWrapper(train_data)
    valid_data = HFDatasetWrapper(valid_data)

    if sample_size and sample_size < len(train_data):
        indices = random.sample(range(len(train_data)), sample_size)
        train_data = Subset(train_data, indices)

    train_sampler = DistributedSampler(train_data) if is_distributed else None
    valid_sampler = DistributedSampler(valid_data) if is_distributed else None

    train_dataloader = tqdm(
        DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collate_fn,
        ),
        desc="train",
    )
    valid_dataloader = tqdm(
        DataLoader(
            valid_data,
            batch_size=batch_size,
            shuffle=(valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=collate_fn,
        ),
        desc="valid",
    )

    return train_dataloader, valid_dataloader


def main():
    vocab_path = "ne_en_vocab.pt"
    vocab_src, vocab_tgt = load_vocab(load_fn=load_ne_en, vocab_path=vocab_path)
    pad_id = vocab_src["<blank>"]

    train_loader, val_loader = create_dataloaders(
        device="cpu",
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        load_fn=load_ne_en,
        batch_size=256,
        max_padding=256,
        is_distributed=False,
    )

    diagnose_tokenization(train_loader, vocab_src, vocab_tgt, pad_id)

    train_src_stats, train_tgt_stats = compute_length_stats(
        train_loader, pad_id, sample_size=10_000, desc="train"
    )
    val_src_stats, val_tgt_stats = compute_length_stats(
        val_loader, pad_id, sample_size=999_999, desc="val"
    )

    print("\n Train SRC Length Stats ", train_src_stats)
    print(" Train TGT Length Stats ", train_tgt_stats)
    print("\n Valid SRC Length Stats ", val_src_stats)
    print(" Valid TGT Length Stats ", val_tgt_stats)

    recommended_src_len = max(train_src_stats["95p"], val_src_stats["95p"]) + 2
    recommended_tgt_len = max(train_tgt_stats["95p"], val_tgt_stats["95p"]) + 2

    print("SRC max_padding:", recommended_src_len)
    print("TGT max_padding:", recommended_tgt_len)

    return {
        "train_src": train_src_stats,
        "train_tgt": train_tgt_stats,
        "val_src": val_src_stats,
        "val_tgt": val_tgt_stats,
        "best_src_len": recommended_src_len,
        "best_tgt_len": recommended_tgt_len,
    }


if __name__ == "__main__":
    main()

import torch
from sst2 import SST2Dataset
import numpy as np


def validate_sst2_dataset(dataset, split_name="train", num_samples=10):
    print(f"\n=== {split_name.upper()} DATASET VALIDATION ===")

    # 1. Basic dataset info
    print(f"Dataset size: {len(dataset)}")
    print(f"Vocab size: {dataset.vocab_size}")
    print(f"Sequence length: {dataset.seq_len}")
    print(f"Number of classes: {dataset.num_classes}")

    # 2. Label distribution - CRITICAL CHECK
    label_dist = dataset.get_label_distribution()
    print(f"\nLabel distribution: {label_dist}")

    # Check for severe class imbalance
    labels = torch.tensor(dataset.labels)
    unique_labels, counts = torch.unique(labels, return_counts=True)
    print(f"Unique labels: {unique_labels.tolist()}")
    print(f"Label counts: {counts.tolist()}")

    # Calculate class balance ratio
    if len(counts) == 2:
        ratio = min(counts) / max(counts)
        print(
            f"""Class balance ratio: {ratio:.3f}
                (should be > 0.3 for good training)"""
        )
        if ratio < 0.1:
            print("⚠️  WARNING: Severe class imbalance detected!")

    # 3. Sample data inspection
    print(f"\n=== SAMPLE DATA INSPECTION ===")
    for i in range(min(num_samples, len(dataset))):
        encoded_sentence, label = dataset[i]

        # Decode sentence back to text
        decoded_tokens = []
        for token_id in encoded_sentence:
            if token_id.item() == dataset.vocab["<pad>"]:
                break
            # Find token by value (reverse lookup)
            token = next(
                (k for k, v in dataset.vocab.items() if v == token_id.item()),
                f"UNK_{token_id.item()}",
            )
            decoded_tokens.append(token)

        decoded_sentence = " ".join(decoded_tokens)
        original_sentence = dataset.sentences[i]

        print(f"\nSample {i + 1}:")
        print(f"  Original: {original_sentence}")
        print(f"  Decoded:  {decoded_sentence}")
        print(
            f"""  Label: {label.item()}
                                     ({"Positive" if label.item() == 1 else "Negative"})"""
        )
        print(f"  Encoded shape: {encoded_sentence.shape}")

    # 4. Vocabulary inspection
    print(f"\n=== VOCABULARY INSPECTION ===")
    print(
        f"""Special tokens: <pad>={dataset.vocab["<pad>"]}, <unk>={
            dataset.vocab["<unk>"]
        }, <eos>={dataset.vocab["<eos>"]}"""
    )

    # Most common tokens
    vocab_by_freq = sorted(
        [
            (k, v)
            for k, v in dataset.vocab.items()
            if k not in ["<pad>", "<unk>", "<eos>"]
        ],
        key=lambda x: x[1],
    )[:20]
    print(f"First 20 tokens by ID: {vocab_by_freq}")

    # 5. Sequence length analysis
    print(f"\n=== SEQUENCE LENGTH ANALYSIS ===")
    sentence_lengths = [len(sent.split()) for sent in dataset.sentences]
    print(f"Sentence length stats:")
    print(f"  Min: {min(sentence_lengths)}")
    print(f"  Max: {max(sentence_lengths)}")
    print(f"  Mean: {np.mean(sentence_lengths):.2f}")
    print(f"  Median: {np.median(sentence_lengths):.2f}")

    # Check for padding issues
    encoded_lengths = []
    for encoded_sentence, _ in dataset:
        # Count non-padding tokens
        non_pad_count = (encoded_sentence != dataset.vocab["<pad>"]).sum().item()
        encoded_lengths.append(non_pad_count)

    print(f"Encoded length stats (non-padding tokens):")
    print(f"  Min: {min(encoded_lengths)}")
    print(f"  Max: {max(encoded_lengths)}")
    print(f"  Mean: {np.mean(encoded_lengths):.2f}")

    # 6. Check for data corruption
    print(f"\n=== DATA CORRUPTION CHECKS ===")

    # Check for empty sentences
    empty_sentences = sum(1 for sent in dataset.sentences if not sent.strip())
    print(f"Empty sentences: {empty_sentences}")

    # Check for invalid labels
    invalid_labels = sum(1 for label in dataset.labels if label not in [0, 1])
    print(f"Invalid labels (not 0 or 1): {invalid_labels}")

    # Check encoding consistency
    encoding_errors = 0
    for i, (encoded_sentence, _) in enumerate(dataset):
        if len(encoded_sentence) != dataset.seq_len:
            encoding_errors += 1
            if encoding_errors == 1:  # Only print first error
                print(
                    f"""  First encoding error at index {i}: length {
                        len(encoded_sentence)
                    } != {dataset.seq_len}"""
                )

    print(f"Encoding length errors: {encoding_errors}")

    # 7. Parse tree validation (spot check)
    print(f"\n=== PARSE TREE VALIDATION ===")
    if hasattr(dataset, "_load_data"):
        try:
            # Test a few parse tree examples manually
            test_trees = [
                "(3 (2 The) (4 (2 Rock) (2 is) (4 (3 destined) (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 century) (2 's) (2 (2 new) (3 (2 conan) (2) (2 and) (2 (2 that) (2 (2 he) (2 's) (2 (2 going) (2 (2 to) (2 (2 make) (2 (2 a) (2 (2 splash) (2 (2 even) (3 greater))))))))))))))))))",
                "(1 (1 (1 Unflinchingly) (2 bleak)) (1 (1 and) (0 (1 desperate) (0 .))))",
            ]

            for tree in test_trees:
                try:
                    sentence, label = dataset._parse_tree(tree)
                    print(f"  Parsed: '{sentence}' -> Label: {label}")
                except Exception as e:
                    print(f"  Parse error: {e}")

        except Exception as e:
            print(f"Parse tree validation failed: {e}")

    return {
        "dataset_size": len(dataset),
        "vocab_size": dataset.vocab_size,
        "label_distribution": label_dist,
        "class_balance_ratio": ratio if len(counts) == 2 else None,
        "empty_sentences": empty_sentences,
        "invalid_labels": invalid_labels,
        "encoding_errors": encoding_errors,
    }


def check_dataloader_batch(dataloader, dataset_name=""):
    """Check if dataloader produces correct batches"""
    print(f"\n=== {dataset_name} DATALOADER BATCH CHECK ===")

    batch = next(iter(dataloader))
    sentences, labels = batch

    print(f"Batch shape - Sentences: {sentences.shape}, Labels: {labels.shape}")
    print(f"Label distribution in batch: {torch.bincount(labels)}")
    print(f"Sample sentence tensor: {sentences[0]}")
    print(f"Sample label: {labels[0].item()}")

    return batch


# Usage example:


def run_full_validation(cfg):
    """Run complete dataset validation"""

    # Load datasets
    train_dataset = SST2Dataset(cfg, "train")
    vocab = train_dataset.get_vocab()
    valid_dataset = SST2Dataset(cfg, "valid", vocab=vocab)

    # Validate datasets
    train_stats = validate_sst2_dataset(train_dataset, "train")
    valid_stats = validate_sst2_dataset(valid_dataset, "valid", num_samples=5)

    # Check dataloaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    train_batch = check_dataloader_batch(train_loader, "TRAIN")
    valid_batch = check_dataloader_batch(valid_loader, "VALID")

    # Summary
    print(f"\n=== VALIDATION SUMMARY ===")
    issues = []

    if train_stats.get("class_balance_ratio", 1) < 0.3:
        issues.append("Poor class balance")
    if train_stats.get("empty_sentences", 0) > 0:
        issues.append("Empty sentences found")
    if train_stats.get("invalid_labels", 0) > 0:
        issues.append("Invalid labels found")
    if train_stats.get("encoding_errors", 0) > 0:
        issues.append("Encoding length errors")

    if issues:
        print(f"⚠️  Issues found: {', '.join(issues)}")
    else:
        print("✅ Dataset validation passed!")

    return train_stats, valid_stats


if __name__ == "__main__":
    cfg = {
        "data_dir": "sst2_data/",
        "url": "your_sst2_url",
        "seq_len": 128,
        "batch_size": 32,
    }

    train_stats, valid_stats = run_full_validation(cfg)

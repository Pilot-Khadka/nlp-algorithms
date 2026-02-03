import json

from util.util import convert_to_attrdict
from dataset.downloader import TatoebaDownloader
from trans_encoder_decoder.train import (
    train,
    setup_distributed,
    is_main_process,
    cleanup_distributed,
)


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
            "batch_size": 16,
            "num_epochs": 50,
            "learning_rate": 1e-3,
            "checkpoint_dir": "checkpoint",
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

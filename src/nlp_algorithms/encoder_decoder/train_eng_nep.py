import json

from nlp_algorithms.util.general_util import convert_to_attrdict


def main():
    from nlp_algorithms.encoder_decoder.train import (
        setup_distributed,
        is_main_process,
        cleanup_distributed,
        train,
    )

    cfg = {
        "dataset": {
            "name": "huggingface",
            "name2": "eng_nep",
            "data_dir": "../dataset/dataset_hf_eng_nep",
            "repos": [
                {
                    "id": "sharad461/ne-en-parallel-208k",
                    "files": "*",  # or omit "files" to download all
                },
                {
                    "id": "openlanguagedata/flores_plus",
                    "files": ["dev/npi_Deva.jsonl", "devtest/npi_Deva.jsonl"],
                },
            ],
            "vocab_size": 10000,
            "sequence_length": 80,
            "max_samples": 1000,
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

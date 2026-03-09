if __name__ == "__main__":
    from nlp_algorithms.dataset.downloader import TatoebaDownloader
    from nlp_algorithms.dataset.reader import TatoebaDataset
    from nlp_algorithms.tokenization import WhitespaceTokenizer
    from nlp_algorithms.infra.vocabulary import Vocabulary

    cfg = {
        "dataset": {
            "url": "https://object.pouta.csc.fi/Tatoeba-Challenge-v2023-09-26/eng-nep.tar",
            "data_dir": "../../dataset/dataset_tatoeba_eng_nep/",
            "vocab_size": 10000,
        },
    }

    data_dir = TatoebaDownloader.download_and_prepare(cfg)
    test_dataset = TatoebaDataset(data_dir, split="test")

    example = test_dataset[0]
    print(example)

    sample = test_dataset[0]
    source_text = sample["src"]
    target_text = sample["tgt"]

    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(source_text)
    print("tokens source:", tokenizer.tokenize(source_text))
    print("tokens target:", tokenizer.tokenize(target_text))

    vocab = Vocabulary()
    # pyrefly: ignore [missing-attribute]
    vocab = Vocabulary.from_tokens(tokens=tokens, vocab_size=10000, min_freq=1)

    print("Vocab size:", len(vocab))
    print("ID of 'the':", vocab.token_to_id.get("the"))
    print("First 10 vocab items:", list(vocab.token_to_id.items())[:10])

    encoded = vocab.encode(tokens[:20])
    decoded = vocab.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)

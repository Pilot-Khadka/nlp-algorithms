if __name__ == "__main__":
    from dataset.downloader import ImdbDownloader
    from dataset.reader import IMDBDataset
    from core_tokenization import WhitespaceTokenizer

    cfg = {
        "dataset": {
            "data_dir": "../../dataset/dataset_imdb/",
            "url": "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
            "archive_name": "aclImdb_v1.tar.gz",
        }
    }

    data_dir = ImdbDownloader.download_and_prepare(cfg)
    test_dataset = IMDBDataset(data_dir, split="test")

    example = test_dataset[0]
    print(example)

    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(example["text"])
    print("tokens:", tokens)

    from infra.vocabulary import Vocabulary

    vocab = Vocabulary()
    vocab = Vocabulary.from_tokens(tokens=tokens, vocab_size=10000, min_freq=1)

    print("Vocab size:", len(vocab))
    print("ID of 'the':", vocab.token_to_id.get("the"))
    print("First 10 vocab items:", list(vocab.token_to_id.items())[:10])

    encoded = vocab.encode(tokens[:20])
    decoded = vocab.decode(encoded)

    print("Encoded:", encoded)
    print("Decoded:", decoded)

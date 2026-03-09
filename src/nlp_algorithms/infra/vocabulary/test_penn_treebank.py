from nlp_algorithms.dataset.reader import PTBDataset

if __name__ == "__main__":
    from nlp_algorithms.dataset.downloader import PTBDownloader
    from nlp_algorithms.tokenization import WhitespaceTokenizer
    from nlp_algorithms.infra.vocabulary import Vocabulary

    cfg = {
        "dataset": {
            "data_dir": "../../dataset/dataset_penn_treebank/",
            "train_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt",
            "valid_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt",
            "test_url": "https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt",
        }
    }

    data_dir = PTBDownloader.download_and_prepare(cfg)
    test_dataset = PTBDataset(data_dir, split="test")

    sample = test_dataset[0]
    raw_text = sample["text"]

    tokenizer = WhitespaceTokenizer()
    tokens = tokenizer.tokenize(raw_text)
    print("tokens:", tokens)

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

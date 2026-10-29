from nlp_algorithms.encoder_decoder.train import overfit_subset
from nlp_algorithms.encoder_decoder.data import load_vocab, load_ne_en


if __name__ == "__main__":
    vocab_src, vocab_tgt = load_vocab(
        load_fn=load_ne_en,
        vocab_path="checkpoint/ne_en/vocab.pt",
    )
    overfit_subset(
        vocab_src=vocab_src,
        vocab_tgt=vocab_tgt,
        load_fn=load_ne_en,
        n_epochs=100,
        batch_size=16,
        max_padding=72,
    )

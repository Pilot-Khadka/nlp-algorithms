from nlp_algorithms.encoder_decoder.train import overfit_subset
from nlp_algorithms.encoder_decoder.data import load_vocab


if __name__ == "__main__":
    vocab_src, vocab_tgt = load_vocab()
    overfit_subset(vocab_src, vocab_tgt)

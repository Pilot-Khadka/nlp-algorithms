import torch.nn as nn

from util.model_loading import load_model_from_name


class EmbeddingFactory:
    @staticmethod
    def create(cfg_model, vocab):
        vocab_size = len(vocab)
        dim = cfg_model.input_dim

        if getattr(cfg_model, "use_pretrained_embedding", False):
            return EmbeddingFactory._load_pretrained(cfg_model, vocab, dim)

        return nn.Embedding(vocab_size, dim)

    @staticmethod
    def _load_pretrained(cfg_model, vocab, dim):
        if cfg_model.embedding_path is None:
            raise ValueError("embedding_path required for pretrained embeddings")

        w2v = load_model_from_name(cfg_model.embedding_path)
        weights = w2v.get_input_embeddings().data

        emb = nn.Embedding(len(vocab), dim)

        # load overlapping region
        size = min(len(vocab), weights.shape[0])
        emb.weight[:size].copy_(weights[:size])

        if len(vocab) > weights.shape[0]:
            nn.init.normal_(emb.weight[weights.shape[0] :])

        return emb

class DatasetBundle:
    def __init__(
        self,
        train_loader,
        valid_loader,
        test_loader,
        vocab=None,
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.vocab = vocab

    @property
    def vocab_size(self):
        return self.train_loader.dataset.vocab_size

    @property
    def num_classes(self):
        return self.train_loader.dataset.num_classes

class DatasetBundle:
    def __init__(
        self, train_loader, valid_loader, test_loader, vocab=None, label_set=None
    ):
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.vocab = vocab
        self.label_set = label_set

    @property
    def vocab_size(self):
        return len(self.vocab) if self.vocab else None

    @property
    def num_classes(self):
        return len(self.label_set) if self.label_set else None

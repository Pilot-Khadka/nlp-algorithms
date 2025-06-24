import os
import torch


DATA_DIR = "./ptb_data"


def read_tokens(split):
    file_path = os.path.join(DATA_DIR, f"{split}.txt")
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().replace("\n", " <eos> ").split()


train_tokens = read_tokens("train")
valid_tokens = read_tokens("valid")
test_tokens = read_tokens("test")


class Vocab:
    def __init__(self, tokens):
        self.itos = list(sorted(set(tokens)))
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

    def encode(self, tokens):
        return [self.stoi[token] for token in tokens]

    def decode(self, indices):
        return [self.itos[i] for i in indices]


vocab = Vocab(train_tokens)
vocab_size = len(vocab.itos)
print("Vocab size:", vocab_size)

train_ids = torch.tensor(vocab.encode(train_tokens), dtype=torch.long)
valid_ids = torch.tensor(vocab.encode(valid_tokens), dtype=torch.long)
test_ids = torch.tensor(vocab.encode(test_tokens), dtype=torch.long)


def batchify(data, batch_size):
    n_batch = data.size(0) // batch_size
    data = data[: n_batch * batch_size]
    # shape: [num_steps, batch_size]
    return data.view(batch_size, -1).t().contiguous()


batch_size = 32
train_data = batchify(train_ids, batch_size)  # e.g., [num_steps, 32]


def get_batch(source, i, seq_len):
    seq_len = min(seq_len, len(source) - 1 - i)
    x = source[i : i + seq_len]
    y = source[i + 1 : i + 1 + seq_len]
    return x, y


def main():
    seq_len = 30
    x, y = get_batch(train_data, 0, seq_len)
    print("Input:", x.shape)  # [seq_len, batch_size]
    print("Target:", y.shape)


if __name__ == "__main__":
    main()

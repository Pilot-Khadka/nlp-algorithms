import torch
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

random.seed(0)

g = torch.Generator().manual_seed(0)

with open("../names.txt", "r") as f:
    words = f.read().splitlines()

print(words[:10])


chars = sorted(list(set("".join(words))))
stoi = {s: i + 1 for i, s in enumerate(chars)}
# also add '.' as index 0
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}
vocab_size = len(chars) + 1
print(stoi)


# build the dataset

block_size = 3  # context length


def build_dataset(words: list) -> tuple:
    X, Y = [], []
    for w in words:
        # print(w)
        context = [0] * block_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print(''.join(itos[i] for i in context),'---->',itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


random.shuffle(words)
n1 = int(len(words) * 0.8)
n2 = int(len(words) * 0.9)

X_train, Y_train = build_dataset(words[:n1])
X_val, Y_val = build_dataset(words[n1:n2])
X_test, Y_test = build_dataset(words[n2:])


n_embed = 10
n_hidden = 200
C = torch.randn((27, n_embed), generator=g)
W1 = torch.randn((30, n_hidden), generator=g)
b1 = torch.randn(n_hidden, generator=g)
W2 = torch.randn((200, vocab_size), generator=g)
b2 = torch.randn(vocab_size, generator=g)
parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True

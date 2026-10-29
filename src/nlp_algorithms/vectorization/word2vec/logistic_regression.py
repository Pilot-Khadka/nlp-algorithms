import torch
import torch.nn as nn
import torch.optim as optim


class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # input embeddings: center words
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        # output embeddings: context words
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

        nn.init.uniform_(self.in_embed.weight, -1.0, 1.0)
        nn.init.uniform_(self.out_embed.weight, -1.0, 1.0)

    def forward(self, center_words, context_words, labels):
        # shape: (B, D)
        center_embeds = self.in_embed(center_words)
        context_embeds = self.out_embed(context_words)

        scores = torch.sum(center_embeds * context_embeds, dim=1)
        probs = torch.sigmoid(scores)

        loss = nn.functional.binary_cross_entropy(probs, labels.float())
        return loss


def train_skipgram(model, dataset, epochs=10, lr=0.01, batch_size=4):
    optimizer = optim.SGD(model.parameters(), lr=lr)
    dataset_tensor = torch.tensor(dataset, dtype=torch.long)

    for epoch in range(epochs):
        total_loss = 0
        perm = torch.randperm(len(dataset_tensor))
        dataset_tensor = dataset_tensor[perm]

        for i in range(0, len(dataset_tensor), batch_size):
            batch = dataset_tensor[i : i + batch_size]
            center, context, label = batch[:, 0], batch[:, 1], batch[:, 2]

            optimizer.zero_grad()
            loss = model(center, context, label)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)

        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")


def main():
    dataset = [
        (10, 20, 1),
        (15, 30, 0),
        (25, 40, 1),
        (35, 50, 0),
    ]
    vocab_size = 100
    embedding_dim = 50

    model = SkipGramNS(vocab_size, embedding_dim)
    train_skipgram(model, dataset, epochs=20, lr=0.05)
    print("Sample embedding for word 10:", model.in_embed.weight[10])


if __name__ == "__main__":
    main()

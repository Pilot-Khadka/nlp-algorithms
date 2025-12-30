"""
Neural Probabilistic Language Models: Insights from Yoshua Bengio
Building Makemore: Insights from Andrej Karpathy
Watch the video: https://www.youtube.com/watch?v=TCH_1BHY58I&t=2983s

This implementation uses a character-level approach with a fixed context size of 3 characters.
Each word in the training data is transformed into a compact feature vector based on its
3-character sequences, diverging from the original paper's table look-up method.
"""

import torch
import torch.nn.functional as F
import random
from pathlib import Path
from typing import Tuple, List


class CharacterLevelMLP:
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 3,
        n_embed: int = 10,
        n_hidden: int = 200,
        seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embed = n_embed
        self.n_hidden = n_hidden

        self.generator = torch.Generator().manual_seed(seed)

        self._init_parameters()

    def _init_parameters(self):
        self.C = torch.randn((self.vocab_size, self.n_embed), generator=self.generator)

        input_size = self.block_size * self.n_embed
        self.W1 = torch.randn((input_size, self.n_hidden), generator=self.generator)
        self.b1 = torch.randn(self.n_hidden, generator=self.generator)

        self.W2 = torch.randn(
            (self.n_hidden, self.vocab_size), generator=self.generator
        )
        self.b2 = torch.randn(self.vocab_size, generator=self.generator)

        self.parameters = [self.C, self.W1, self.b1, self.W2, self.b2]

        for p in self.parameters:
            p.requires_grad = True

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        emb = self.C[X]  # (batch_size, block_size, n_embed)

        # (batch_size, block_size * n_embed)
        emb_flat = emb.view(-1, self.W1.shape[0])

        h = torch.tanh(emb_flat @ self.W1 + self.b1)  # (batch_size, n_hidden)

        logits = h @ self.W2 + self.b2  # (batch_size, vocab_size)

        return logits

    def calculate_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(X)
        return F.cross_entropy(logits, Y)

    def get_num_parameters(self) -> int:
        return sum(p.nelement() for p in self.parameters)


class LanguageModelTrainer:
    def __init__(self, model: CharacterLevelMLP):
        self.model = model

    def find_learning_rate(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        lr_range: Tuple[float, float] = (-3, 0),
        iterations: int = 1000,
        batch_size: int = 32,
    ) -> Tuple[List, List, List]:
        lre = torch.linspace(lr_range[0], lr_range[1], iterations)
        lrs = 10**lre

        param_copies = []
        for p in self.model.parameters:
            p_copy = p.clone().detach()
            p_copy.requires_grad = True
            param_copies.append(p_copy)

        lri, lossi, stepi = [], [], []

        for i in range(iterations):
            ix = torch.randint(0, X_train.shape[0], (batch_size,))

            emb = param_copies[0][X_train[ix]]
            h = torch.tanh(
                emb.view(-1, param_copies[1].shape[0]) @ param_copies[1]
                + param_copies[2]
            )
            logits = h @ param_copies[3] + param_copies[4]
            loss = F.cross_entropy(logits, Y_train[ix])

            for p in param_copies:
                p.grad = None
            loss.backward()

            lr = lrs[i]
            for p in param_copies:
                p.data -= lr * p.grad

            lri.append(lr.item())
            lossi.append(loss.item())
            stepi.append(i)

        return lri, lossi, stepi

    def train(
        self,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        iterations: int = 200000,
        batch_size: int = 32,
        lr_schedule: Tuple[float, float, int] = (0.1, 0.01, 100000),
    ) -> Tuple[List, List]:
        lossi, stepi = [], []
        initial_lr, final_lr, switch_point = lr_schedule

        for i in range(iterations):
            ix = torch.randint(0, X_train.shape[0], (batch_size,))
            loss = self.model.calculate_loss(X_train[ix], Y_train[ix])

            if i % 10000 == 0:
                print(f"Iteration {i:6d}/{iterations}: Loss = {loss.item():.4f}")

            for p in self.model.parameters:
                p.grad = None
            loss.backward()

            lr = initial_lr if i < switch_point else final_lr
            for p in self.model.parameters:
                p.data -= lr * p.grad

            stepi.append(i)
            lossi.append(loss.log10().item())

        return lossi, stepi

    def evaluate(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        with torch.no_grad():
            loss = self.model.calculate_loss(X, Y)
        return loss.item()


class DataProcessor:
    def __init__(self, words: List[str], block_size: int = 3):
        self.words = words
        self.block_size = block_size

        self.chars = sorted(list(set("".join(words))))
        self.stoi = {s: i + 1 for i, s in enumerate(self.chars)}
        self.stoi["."] = 0  # End-of-word token
        self.itos = {i: s for s, i in self.stoi.items()}
        self.vocab_size = len(self.chars) + 1

    def build_dataset(self, words: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        X, Y = [], []

        for word in words:
            # Initialize with end-of-word tokens
            context = [0] * self.block_size
            for ch in word + ".":  # Add end-of-word token
                ix = self.stoi[ch]
                X.append(context.copy())
                Y.append(ix)
                context = context[1:] + [ix]  # Shift context

        return torch.tensor(X), torch.tensor(Y)

    def split_data(
        self, train_ratio: float = 0.8, val_ratio: float = 0.1
    ) -> Tuple[torch.Tensor, ...]:
        words_shuffled = self.words.copy()
        random.shuffle(words_shuffled)

        n1 = int(len(words_shuffled) * train_ratio)
        n2 = int(len(words_shuffled) * (train_ratio + val_ratio))

        X_train, Y_train = self.build_dataset(words_shuffled[:n1])
        X_val, Y_val = self.build_dataset(words_shuffled[n1:n2])
        X_test, Y_test = self.build_dataset(words_shuffled[n2:])

        return X_train, Y_train, X_val, Y_val, X_test, Y_test


class TextGenerator:
    def __init__(self, model: CharacterLevelMLP, itos: dict, block_size: int):
        self.model = model
        self.itos = itos
        self.block_size = block_size

    def generate_samples(self, num_samples: int = 20, seed: int = 0) -> List[str]:
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
        else:
            g = torch.Generator()

        samples = []

        for _ in range(num_samples):
            out = []
            context = [0] * self.block_size

            while True:
                with torch.no_grad():
                    logits = self.model.forward(torch.tensor([context]))
                    probs = F.softmax(logits, dim=1)
                    ix = torch.multinomial(probs, num_samples=1, generator=g).item()

                context = context[1:] + [ix]
                out.append(ix)

                if ix == 0:
                    break

            sample = "".join(self.itos[i] for i in out[:-1])
            samples.append(sample)

        return samples


def visualize_embeddings(
    C: torch.Tensor, itos: dict, figsize: Tuple[int, int] = (8, 8)
):
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.scatter(C[:, 0].data, C[:, 1].data, s=200)
    for i in range(C.shape[0]):
        plt.text(
            C[i, 0].item(),
            C[i, 1].item(),
            itos[i],
            ha="center",
            va="center",
            color="white",
        )
    plt.grid("minor")
    plt.title("Character Embeddings Visualization")
    plt.xlabel("Embedding Dimension 0")
    plt.ylabel("Embedding Dimension 1")


def main():
    import matplotlib.pyplot as plt

    random.seed(0)
    torch.manual_seed(0)

    data_path = Path("names.txt")
    if not data_path.exists():
        print(f"file {data_path} not found.")
        return

    with open(data_path, "r") as f:
        words = f.read().splitlines()

    print(f"Loaded {len(words)} words")

    processor = DataProcessor(words, block_size=3)
    print(f"Vocabulary size: {processor.vocab_size}")

    X_train, Y_train, X_val, Y_val, X_test, Y_test = processor.split_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    model = CharacterLevelMLP(
        vocab_size=processor.vocab_size, block_size=3, n_embed=10, n_hidden=200, seed=0
    )

    trainer = LanguageModelTrainer(model)
    print("\nFinding optimal learning rate...")
    lri, lossi, stepi = trainer.find_learning_rate(X_train, Y_train, iterations=1000)

    plt.figure(figsize=(10, 6))
    plt.plot(lri, lossi)
    plt.xscale("log")
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.grid(True)
    plt.show()

    print("\nTraining model...")
    losses, steps = trainer.train(X_train, Y_train, iterations=200000)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Log10 Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.show()

    print("\nEvaluating model...")
    train_loss = trainer.evaluate(X_train, Y_train)
    val_loss = trainer.evaluate(X_val, Y_val)
    test_loss = trainer.evaluate(X_test, Y_test)

    print(f"Training loss: {train_loss:.4f}")
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")

    if model.n_embed >= 2:
        print("\nVisualizing character embeddings...")
        visualize_embeddings(model.C, processor.itos)
        plt.show()

    print("\nGenerating samples...")
    generator = TextGenerator(model, processor.itos, processor.block_size)
    samples = generator.generate_samples(num_samples=20, seed=2147483647 + 10)

    print("Generated names:")
    for i, sample in enumerate(samples, 1):
        print(f"{i:2d}. {sample}")


if __name__ == "__main__":
    main()

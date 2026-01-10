from typing import List, Tuple


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from a07_lstm.lstm import LSTM


class RepeatingPatternDataset(Dataset):
    def __init__(
        self,
        pattern: List[int],
        num_sequences: int,
        seq_length: int,
        seed: int = 42,
    ):
        self.pattern = pattern
        self.pattern_len = len(pattern)
        self.num_sequences = num_sequences
        self.seq_length = seq_length

        torch.manual_seed(seed)
        self.sequences = []

        for _ in range(num_sequences):
            start_idx = int(torch.randint(0, self.pattern_len, (1,)).item())
            seq = []
            for i in range(seq_length):
                seq.append(pattern[(start_idx + i) % self.pattern_len])
            self.sequences.append(seq)

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[index]
        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)
        return x, y


def train_epoch(
    model: LSTM,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    num_batches = 0

    model.reset_state()

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        output, _ = model(x)

        loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(
    model: LSTM,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    num_batches = 0

    model.reset_state()

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            output, _ = model(x)

            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()

            predictions = output.argmax(dim=-1)
            correct += (predictions == y).sum().item()
            total += y.numel()
            num_batches += 1

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy


def generate_sequence(
    model: LSTM,
    start_token: int,
    length: int,
    device: torch.device,
) -> List[int]:
    model.eval()
    model.reset_state()

    generated = [start_token]
    current = torch.tensor([[start_token]], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(length - 1):
            output, _ = model(current)
            next_token = output[0, -1].argmax().item()
            generated.append(next_token)
            current = torch.tensor([[next_token]], dtype=torch.long, device=device)

    return generated


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pattern = [1, 2, 3, 4, 5]
    vocab_size = max(pattern) + 1

    train_dataset = RepeatingPatternDataset(
        pattern=pattern,
        num_sequences=1000,
        seq_length=50,
        seed=42,
    )
    test_dataset = RepeatingPatternDataset(
        pattern=pattern,
        num_sequences=200,
        seq_length=50,
        seed=123,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = LSTM(
        input_dim=128,
        hidden_dim=256,
        output_dim=vocab_size,
        num_layers=2,
        dropout=0.3,
        embed_dropout=0.1,
        recurrent_dropout=0.2,
        output_dropout=0.3,
        tie_weights=False,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 50
    train_losses = []
    test_losses = []
    test_accuracies = []

    print(f"\nTraining pattern: {pattern}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequences: {len(train_dataset)}")
    print(f"Test sequences: {len(test_dataset)}\n")

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Test Loss: {test_loss:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")

    print("\n" + "=" * 50)
    print("Generation Test")
    print("=" * 50)

    for start_token in pattern[:3]:
        generated = generate_sequence(model, start_token, 20, device)
        print(f"Start: {start_token}, Generated: {generated}")


if __name__ == "__main__":
    main()

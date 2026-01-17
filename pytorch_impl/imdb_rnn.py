from typing import Dict, Tuple

import torch
import torch.nn as nn

from tqdm import tqdm


class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.5,
        bidirectional: bool = True,
        rnn_type: str = "lstm",
        num_classes: int = 2,
        pad_idx: int = 0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        rnn_class = {
            "lstm": nn.LSTM,
            "gru": nn.GRU,
            "rnn": nn.RNN,
        }[self.rnn_type]

        self.rnn = rnn_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(rnn_output_dim, num_classes)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(input_ids))

        if self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]

        hidden = self.dropout(hidden)
        logits = self.fc(hidden)

        return logits


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for input_ids, labels, doc_ids, chunk_ids in pbar:
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
        )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def evaluate(
    model: nn.Module,
    eval_loader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Evaluating",
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(eval_loader, desc=desc, leave=False)
        for input_ids, labels, doc_ids, chunk_ids in pbar:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)

            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{100 * correct / total:.2f}%"}
            )

    avg_loss = total_loss / len(eval_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int,
    learning_rate: float,
    device,
) -> Dict[str, list]:
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, desc="Validation"
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.2f}%")

    return history


def test_model(
    model: nn.Module,
    test_loader,
    device,
) -> Tuple[float, float]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    print("\nTesting model...")
    print("-" * 50)

    test_loss, test_acc = evaluate(
        model, test_loader, criterion, device, desc="Testing"
    )

    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")

    return test_loss, test_acc


if __name__ == "__main__":
    from dataset.imdb import get_imdb_dataloaders
    from util.util import load_config

    cfg = load_config("../config/pytorch_lstm_imdb.yaml")
    dataset_bundle = get_imdb_dataloaders(cfg)
    vocab_size = len(dataset_bundle.vocab_size)
    print("vocab size:", vocab_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocabulary size: {vocab_size}")

    model = SentimentRNN(
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2,
        dropout=0.5,
        bidirectional=True,
        rnn_type="lstm",
        num_classes=2,
        pad_idx=0,
    )

    history = train_model(
        model=model,
        train_loader=dataset_bundle.train_loader,
        val_loader=dataset_bundle.valid_loader,
        num_epochs=int(cfg.train["epochs"]),  # pyrefly ignore
        learning_rate=cfg.train.learning_rate,  # pyrefly ignore
        device=device,
    )

    test_loss, test_acc = test_model(
        model=model,
        test_loader=dataset_bundle.test_loader,
        device=device,
    )

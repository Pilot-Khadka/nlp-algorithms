import torch
import torch.nn as nn
import os
from rnn import RNN
from datasets.data_loader import load_dataset
from engine.utils import setup_logging, save_model, load_model
from engine.train_eval import train_epoch, validate_epoch


def main():
    logger = setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    os.makedirs("checkpoints", exist_ok=True)

    logger.info("Loading dataset...")
    train_loader, valid_loader, test_loader, vocab = load_dataset(
        name="ptb", seq_len=30, batch_size=128
    )

    vocab_size = len(vocab)
    print("vocab size:", vocab_size)
    embedding_dim = 100
    hidden_dim = 256
    output_dim = vocab_size  # for language modeling

    logger.info(f"Vocabulary size: {vocab_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Hidden dimension: {hidden_dim}")

    model = RNN(embedding_dim, hidden_dim, output_dim).to(device)
    embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(embedding.parameters())
    )
    criterion = nn.CrossEntropyLoss()

    num_epochs = 30
    best_valid_loss = float("inf")
    start_epoch = 0
    metrics_to_log = ["perplexity"]
    checkpoint_path = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        logger.info("Loading existing checkpoint...")
        try:
            model, embedding, optimizer, start_epoch, best_valid_loss = load_model(
                checkpoint_path, device
            )
            logger.info(
                f"""Resumed epoch {start_epoch}, best valid loss: {
                    best_valid_loss:.4f}"""
            )

        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            logger.info("Starting training from scratch...")

    logger.info("Starting training...")
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")

        train_loss, metrics = train_epoch(
            model,
            embedding,
            train_loader,
            optimizer,
            criterion,
            device,
            logger,
            metrics_fn=metrics_to_log,
        )

        valid_loss, metrics = validate_epoch(
            model,
            embedding,
            valid_loader,
            criterion,
            device,
            logger,
            metrics_fn=metrics_to_log,
        )

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(
                model,
                embedding,
                optimizer,
                epoch + 1,
                valid_loss,
                checkpoint_path,
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
            )

            logger.info(f"""New best model saved! Validation loss: {
                        valid_loss:.4f}""")

        regular_checkpoint = f"checkpoints/epoch_{epoch + 1}.pt"
        save_model(
            model,
            embedding,
            optimizer,
            epoch + 1,
            valid_loss,
            regular_checkpoint,
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        logger.info(
            f"""Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Valid Loss: {
                valid_loss:.4f}"""
        )

    logger.info("Training completed!")
    logger.info(f"Best validation loss: {best_valid_loss:.4f}")


if __name__ == "__main__":
    main()

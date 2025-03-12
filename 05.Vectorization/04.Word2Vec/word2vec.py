import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from skipgram import Skipgram


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)

        nn.init.uniform_(self.input_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)
        nn.init.uniform_(self.output_embeddings.weight, -0.5 / embedding_dim, 0.5 / embedding_dim)

    def forward(self, center_word, context_word):
        center_vector = self.input_embeddings(center_word)      # (Batch, Embedding_dim)
        context_vector = self.output_embeddings(context_word)   # (Batch, Embedding_dim)

        score = torch.sum(center_vector * context_vector, dim=-1)  # (Batch,)
        return torch.sigmoid(score) 

def train(dataset_generator, 
          vocab_size,
          embedding_dim=300,
          lr=0.001,
          epochs=30,
          batch_size=64):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Word2Vec(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_function = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in dataset_generator(batch_size=batch_size):
            num_batches +=1 
            center_words, context_words, labels = zip(*batch)

            center_word = torch.tensor(center_words, dtype=torch.long, device=device)
            context_word = torch.tensor(context_words, dtype=torch.long, device=device)
            label = torch.tensor(labels, dtype=torch.float32, device=device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                prediction = model(center_word, context_word)
                loss = loss_function(prediction, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            sys.stdout.write(f"\rEpoch {epoch+1}/{epochs} | Batch {num_batches}, Loss: {loss.item():.4f}")
            sys.stdout.flush()
        print(f"\nEpoch {epoch+1}, Avg Loss: {total_loss / num_batches:.4f}")
        
def main():
    skipgram = Skipgram()
    train(dataset_generator=skipgram.batch_generator, 
          vocab_size=len(skipgram.vocab), 
          batch_size=1024)


if __name__ == "__main__":    
    main()

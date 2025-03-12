import math
import random
import numpy as np
#from skipgram import Skipgram

class LogisticRegression:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size         = vocab_size
        self.embedding_dim      = embedding_dim
        self.losses             = []
        self.lr = 5e-4
        self._init_embedding() 

    def _init_embedding(self):
        #goal is to maximize dot product between vectors that come together and minimize that dont
        self.input_embedding    = np.random.uniform(-1,1,(self.vocab_size, self.embedding_dim))
        self.output_embedding   = np.random.uniform(-1,1,(self.vocab_size, self.embedding_dim))

    def train(self, dataset, epochs=30):

        for epoch in range(epochs):
            total_loss = 0
            for center_idx, context_idx, label in dataset:
                center_vector = self.input_embedding[center_idx]
                context_vector = self.output_embedding[context_idx]
                score = np.dot(center_vector, context_vector)
                probab = self._sigmoid(score)
                
                loss = - (label * np.log(probab) + (1-label) * np.log(1-probab))
                total_loss += loss

                grad = probab - label
                self.input_embedding[center_idx] -= self.lr * grad * context_vector
                self.output_embedding[context_idx] = self.lr * grad * center_vector
            self.losses.append(total_loss)
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
    def _sigmoid(self, x):
        if x>=0: # stable
            z = np.exp(-x)
            return 1/(1+z)
        else:
            z = np.exp(x)
            return z/(1+z)


def main():
    #model = Skipgram()
    #dataset = model.build_dataset() 

    # input -> length of vocabulary
    # i.e 
    example_dataset = [
        (10, 20, 1),  # Positive pair
        (15, 30, 0),  # Negative sample
        (25, 40, 1),  # Positive pair
        (35, 50, 0)   # Negative sample
    ]
    regression_model = LogisticRegression(vocab_size = 10000,
                               embedding_dim = 300)
    regression_model.train(example_dataset)

main()

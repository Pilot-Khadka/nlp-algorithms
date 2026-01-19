import numpy as np


LR = 5e-5
ITERATION = 10


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost2(w, X, y):
    m = len(y)
    h = sigmoid(X.dot(w))
    cost = -1 / m * (np.sum(np.log(h) + (1 - y) * np.log(1 - h)))
    return cost


def gradient_descent(X, Y, w):
    m = len(Y)
    cost_history = []
    for i in range(ITERATION):
        h = sigmoid(X.dot(w))
        gradient = X.T.dot(h - Y) / m
        print("Gradient shape:", gradient.shape)
        w -= LR * gradient
        cost = cost2(w, X, Y)
        cost_history.append(cost)
        if i % 1000 == 0:
            print(f"iteration {i}, cost:{cost}")

    return w, cost_history


def predict(X, w):
    threshold = 0.5
    output = sigmoid(X.dot(w))
    # pyrefly: ignore [missing-attribute]
    return output >= threshold.astype(int)


def main():
    x = np.arange(10).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    w = np.zeros(x.shape[1])
    print(f"x:{x.shape}, y:{y.shape}, w:{w.shape}")
    w, costs = gradient_descent(x, y, w)
    print(w, costs)

    # x = [
    # (10, 20, 1),  # Positive pair
    # (15, 30, 0),  # Negative sample
    # (25, 40, 1),  # Positive pair
    # (35, 50, 0)   # Negative sample
    # ]
    # w = np.zeros(50)
    # w,costs = gradient_Descent(x[1], kj


main()

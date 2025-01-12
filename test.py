import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


path = os.listdir("../feedforward net/input")
print(path)

train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# train data, y: digits
y_train = train_data['label'].values

# train data, x: your inputs (pixel values) (normalization)
X_train = train_data.drop(columns=['label']).values / 255

# testing data (normalization)
X_test = test_data.drop(columns=['label']).values / 255  # Ensure correct number of features

def relu(x):
    x[x < 0] = 0
    return x

def h(X, W, b):
    '''
    Hypothesis function: simple FNN with 1 hidden layer
    Layer 1: Input
    Layer 2: Hidden Layer
    Layer 3: Output Layer
    '''
    # layer 1 = input layer
    a1 = X
    # layer 1 (input layer) -> layer 2 (hidden layer)
    z1 = np.matmul(X, W[0]) + b[0]
 

    # layer 2 activation
    a2 = relu(z1)
    # layer 2 (hidden layer) -> layer 3 (output layer)
    sigma = softmax(a2, W[1])

    # return your probability distribution 
    return sigma

def softmax(X_in, weights):
    s = np.exp(np.matmul(X_in, weights))
    total = np.sum(s, axis=1).reshape(-1, 1)
    return s / total

def loss(y_pred, y_true):
    '''
    Loss function: cross entropy with an L^2 regularization
    y_true: ground truth, of shape (N, )
    y_pred: prediction made by the model, of shape (N, K)
    N: number of samples in the batch
    K: global variable, number of classes
    '''
    global K
    K = 10
    N = len(y_true)
    y_true_one_hot_vec = (y_true[:, np.newaxis] == np.arange(K))
    loss_sample = (np.log(y_pred) * y_true_one_hot_vec).sum(axis=1)
    return -np.mean(loss_sample)

def backprop(W, b, X, y, alpha=1e-4):
    '''
    Step 1: explicit forward pass h(X;W,b)
    Step 2: backpropagation for dW and db
    '''
    K = 10
    N = X.shape[0]

    ### Step 1:
    # layer 1 = input layer
    a1 = X
    # layer 1 (input layer) -> layer 2 (hidden layer)
    z1 = np.matmul(X, W[0]) + b[0]
    # layer 2 activation
    a2 = relu(z1)

    # layer 2 (hidden layer) -> layer 3 (output layer)
    z2 = np.matmul(a2, W[1])
    s = np.exp(z2)
    total = np.sum(s, axis=1).reshape(-1, 1)
    sigma = s / total

    ### Step 2:
    y_one_hot_vec = (y[:, np.newaxis] == np.arange(K))
    delta2 = (sigma - y_one_hot_vec)
    grad_W1 = np.matmul(a2.T, delta2)

    delta1 = np.matmul(delta2, W[1].T) * (z1 > 0)
    grad_W0 = np.matmul(X.T, delta1)

    dW = [grad_W0 / N + alpha * W[0], grad_W1 / N + alpha * W[1]]
    db = [np.mean(delta1, axis=0)]
    return dW, db

eta = 5e-1
alpha = 1e-6
gamma = 0.99
eps = 1e-3
num_iter = 10
n_H = 256
n = X_train.shape[1]
K = 10

np.random.seed(2)
W = [1e-1 * np.random.randn(n, n_H), 1e-1 * np.random.randn(n_H, K)]
b = [np.random.randn(n_H)]

gW0 = gW1 = gb0 = 1

for i in range(num_iter):
    print(f"Iteration {i+1}")
    dW, db = backprop(W, b, X_train, y_train, alpha)


    gW0 = gamma * gW0 + (1 - gamma) * np.sum(dW[0] ** 2)
    etaW0 = eta / np.sqrt(gW0 + eps)
    W[0] -= etaW0 * dW[0]

    gW1 = gamma * gW1 + (1 - gamma) * np.sum(dW[1] ** 2)
    etaW1 = eta / np.sqrt(gW1 + eps)
    W[1] -= etaW1 * dW[1]

    gb0 = gamma * gb0 + (1 - gamma) * np.sum(db[0] ** 2)
    etab0 = eta / np.sqrt(gb0 + eps)
    b[0] -= etab0 * db[0]


    if i % 500 == 0:
        y_pred = h(X_train, W, b)
        print("Cross-entropy loss after", i + 1, "iterations is {:.8}".format(
            loss(y_pred, y_train)))
        print("Training accuracy after", i + 1, "iterations is {:.4%}".format(
            np.mean(np.argmax(y_pred, axis=1) == y_train)))

        print("gW0={:.4f} gW1={:.4f} gb0={:.4f}\netaW0={:.4f} etaW1={:.4f} etab0={:.4f}"
              .format(gW0, gW1, gb0, etaW0, etaW1, etab0))

        print("|dW0|={:.5f} |dW1|={:.5f} |db0|={:.5f}"
              .format(np.linalg.norm(dW[0]), np.linalg.norm(dW[1]), np.linalg.norm(db[0])), "\n")

        gW0 = gW1 = gb0 = 1

y_pred_final = h(X_train, W, b)
print("Final cross-entropy loss is {:.8}".format(loss(y_pred_final, y_train)))
print("Final training accuracy is {:.4%}".format(np.mean(np.argmax(y_pred_final, axis=1) == y_train)))
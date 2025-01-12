import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time 
from fnn_utils import h, loss, backprop

start_time = time.time() 

train_data = pd.read_csv('../datasets/train.csv')
test_data = pd.read_csv('../datasets/test.csv')

# train data, y: digits
y_train = train_data['label'].values

# train data, x: your inputs (pixel values) (normalization)
X_train = train_data.drop(columns=['label']).values / 255

# testing data (normalization)
X_test = test_data.drop(columns=['label']).values / 255  

# initialize the weights and biases
eta = 5e-1
alpha = 1e-6
gamma = 0.99
eps = 1e-3
# number of iterations
num_iter = 50
# number of hidden units
n_H = 256
n = X_train.shape[1]
# number of classes (digits)
K = 10

np.random.seed(1127825)
W = [1e-1 * np.random.randn(n, n_H), 1e-1 * np.random.randn(n_H, K)]
b = [np.random.randn(n_H)]

gW0 = gW1 = gb0 = 1

for i in range(num_iter):
    
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

    y_pred = h(X_train, W, b)
    print("Training accuracy at iteration: ", i+1, " is {:.4%}".format(np.mean(np.argmax(y_pred, axis=1) == y_train)))

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
# Save the state of the model
try:
  np.savez('../models/fnn_mnist_trained_model.npz', W0=W[0], W1=W[1], b0=b[0])
  print("Model saved")
except:
  print("An exception occurred")

print("Final cross-entropy loss is {:.8}".format(loss(y_pred_final, y_train)))
print("Final training accuracy is {:.4%}".format(np.mean(np.argmax(y_pred_final, axis=1) == y_train)))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

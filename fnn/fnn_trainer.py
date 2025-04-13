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
num_iter = 100
# number of hidden units
n_H = 256
n = X_train.shape[1]
# number of classes (digits)
K = 10

# Initialization: Live Plot (Training Loss)
loss_values = []
plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlim(0, num_iter)
ax.set_ylim(0, 3) 
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Live Training Loss")


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

    # Plot point 
    current_loss = loss(y_pred, y_train)
    loss_values.append(current_loss)
    line.set_data(range(len(loss_values)), loss_values)
    ax.set_ylim(0, max(loss_values) + 0.1)  # Dynamically rescale y-axis
    plt.pause(0.01)
    
    print("Training accuracy at iteration: ", i+1, " is {:.4%}".format(np.mean(np.argmax(y_pred, axis=1) == y_train)))

    if i % 500 == 0:
        y_pred = h(X_train, W, b)
        gW0 = gW1 = gb0 = 1

y_pred_final = h(X_train, W, b)
# Save the state of the model
try:
  np.savez('../models/fnn_mnist_trained_model.npz', W0=W[0], W1=W[1], b0=b[0])
  print("Model saved")
except:
  print("An exception occurred")

# Final outputs
print("Final cross-entropy loss is {:.8}".format(loss(y_pred_final, y_train)))
print("Final training accuracy is {:.4%}".format(np.mean(np.argmax(y_pred_final, axis=1) == y_train)))
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Training Time: {elapsed_time} seconds")

# Plot termination 
plt.ioff()
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time 

def relu(x):
    # if x > 0, return x. Else, return 0 
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

def h_debug(X, W, b):
    # layer 1
    a1 = X
    # print shape of input matrix
    print("Shape of input matrix is: ", a1.shape)
    # layer 1 (input layer) -> layer 2 (hidden layer)
    print("Shape of weight matrix is: ", W[0].shape)
    # print shape of bias matrix
    print("Shape of bias matrix is: ", b[0].shape)
    z1 = np.matmul(X, W[0]) + b[0]
    # print shape of z1
    print("z1 (after first matmul) shape is: ", z1.shape)
    # print first 10 elements of z1
    print("First 10 elements of z1: ", z1[:10])

    # layer 2 activation
    a2 = relu(z1)
    # print shape of a2 (after relu)
    print("relu has been called, shape of resulting array is: ", a2.shape)
    # print first 10 elements of a2
    print("First 10 elements of z1 after relu: ", a2[:10])
    # layer 2 (hidden layer) -> layer 3 (output layer)
    print("Softmax is about to be executed, matmul op with a2 and W[1]")
    # print shape of W[1]
    print("Shape of W[1] is: ", W[1].shape)
    sigma = softmax(a2, W[1])

    # return your probability distribution 
    print("End result after softmax is: ", sigma.shape) # same shape as y_pred and y_train. what is size of y_test? => 10000 images, ah the 80/20 rule? 
    return sigma

def softmax(X_in, weights):
    s = np.exp(np.matmul(X_in, weights))
    total = np.sum(s, axis=1).reshape(-1, 1) # looks across each row and find the sum of that row. 
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
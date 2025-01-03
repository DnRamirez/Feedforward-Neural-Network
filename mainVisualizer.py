import numpy as np
import pandas as pd
import os

# Assume your CSV files are located in the "../feedforward net/input" folder
path = os.listdir("../feedforward net/input")
print("Available files:", path)

# Load the train and test data
train_data = pd.read_csv("../feedforward net/input/train.csv")
test_data = pd.read_csv("../feedforward net/input/test.csv")

# Extract the labels and features from train data
y_train = train_data['label'].values
X_train = train_data.drop(columns=['label']).values / 255  # Normalize pixel values
X_test = test_data.values / 255  # Normalize test data

# Activation function: ReLU
def relu(x): 
    x[x < 0] = 0
    return x

# Hypothesis function
def h(X, W, b): 
    print("Shape of input X:", X.shape)
    
    # Layer 1 -> Layer 2 (input to hidden layer)
    z1 = np.matmul(X, W[0]) + b[0]
    print("Shape after first matrix multiplication (z1):", z1.shape)
    print("First few values of z1:\n", z1[:2])
    
    a2 = relu(z1)
    print("Shape after applying ReLU (a2):", a2.shape)
    print("Note: Relu sets all negative values to 0")
    print("First few values of a2 after ReLU:\n", a2[:2])
    
    # Layer 2 -> Layer 3 (hidden to output layer)
    z2 = np.matmul(a2, W[1])
    print("Shape after second matrix multiplication (z2):", z2.shape)
    print("First few values of z2:\n", z2[:2])
    
    sigma = softmax(a2, W[1])
    
    print("Shape of output probabilities (sigma):", sigma.shape)
    print("First few values of sigma (softmax probabilities):\n", sigma[:2])
    return sigma 
    
    


def softmax(X_in, weights): 
    s = np.exp(np.matmul(X_in, weights))
    total = np.sum(s, axis=1).reshape(-1, 1)
    return s/total

# Define weights and biases
input_size = 784  # Number of input features (pixels)
hidden_units = 128  # Number of neurons in the hidden layer
output_size = 10    # Number of output units (digit classes 0-9)

# Initialize random weights and biases
np.random.seed(0)
W = [
    np.random.randn(input_size, hidden_units),  # Weights between input and hidden layer
    np.random.randn(hidden_units, output_size)  # Weights between hidden and output layer
]

b = [
    np.random.randn(hidden_units),  # Bias for hidden layer
    np.random.randn(output_size)    # Bias for output layer
]

# Run the hypothesis function on the training data
output_probabilities = h(X_train, W, b)

# Display the first few predicted probabilities
print("\nFirst few output probabilities for each class:\n", output_probabilities[:5])

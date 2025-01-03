import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

    return sigma

def h_debug(X, W, b):
    '''
    Hypothesis function: simple FNN with 1 hidden layer
    Layer 1: Input
    Layer 2: Hidden Layer
    Layer 3: Output Layer
    '''
    # layer 1 = input layer
    a1 = X
    print("Shape of input matrix is: ", a1.shape)
    print("First 10 elements of input matrix: ", a1[:10])
    # layer 1 (input layer) -> layer 2 (hidden layer)
    print("Shape of weight matrix is: ", W[0].shape)
    print("Shape of bias matrix is: ", b[0].shape)
    z1 = np.matmul(X, W[0]) + b[0]

    print("z1 (after first matmul) shape is: ", z1.shape)
    print("First 10 elements of z1 prior to relu: ", z1[:10])

    # layer 2 activation
    a2 = relu(z1)
    print("relu has been called, shape of resulting array is: ", a2.shape)
    print("First 10 elements of z1 after relu: ", a2[:10])
    # layer 2 (hidden layer) -> layer 3 (output layer)
    print("Softmax is about to be executed, matmul op with a2 and W[1]")
    print("Shape of W[1] is: ", W[1].shape)
    sigma = softmax(a2, W[1])

    # return your probability distribution 
    print("End result after softmax is: ", sigma.shape) # same shape as y_pred and y_train. what is size of y_test? => 10000 images, ah the 80/20 rule? 
    return sigma

def softmax(X_in, weights):
    s = np.exp(np.matmul(X_in, weights))
    total = np.sum(s, axis=1).reshape(-1, 1)
    return s / total

# Load the state of the model
model = np.load("model.npz")
W = [model['W0'], model['W1']]
b = [model['b0']]
print("Model loaded")

# Load the test data
test_data = pd.read_csv("test.csv")

# Select a test image and ensure it has the correct number of features
test_image = test_data.drop(columns=['label']).iloc[0].values / 255

# Reshape the test image for display
test_image_reshaped = test_image.reshape(28, 28)

# Make a prediction
test_image_flat = test_image.reshape(1, -1)  # Flatten the image for prediction
y_pred = h_debug(test_image_flat, W, b)
print("Predicted probabilities:", y_pred)
print("Shape of y_pred: ", y_pred.shape)
print("Predicted digit:", np.argmax(y_pred))
print("True digit:", test_data['label'].iloc[0])




X_test = test_data.drop(columns=['label']).values / 255  # Ensure correct number of features
test_index = 4  # You can change this index to select a different test case
test_image = X_test[test_index].reshape(28, 28)  # Ensure correct number of features



# plot the sample 
fig = plt.figure
plt.imshow(test_image, cmap='gray')
plt.show()


# Predict the value using the trained model
test_image_flat = X_test[test_index].reshape(1, -1)  # Flatten the image for prediction
predicted_value = np.argmax(h(test_image_flat, W, b))

# True value (assuming you have the true labels for the test set)
true_value = test_data['label'].values[test_index]

# Display the model's guess and the true value
print(f"Model's guess: {predicted_value}")
print(f"True value: {true_value}")
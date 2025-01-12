import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fnn_utils import h

# Load the state of the model
model = np.load('../models/fnn_mnist_trained_model.npz')
W = [model['W0'], model['W1']]
b = [model['b0']]
print("Model loaded")

# Load the test data
test_data = pd.read_csv('../datasets/test.csv')

X_test = test_data.drop(columns=['label']).values / 255  
# You can change this index to select a different test case
test_index = 4 
test_image = X_test[test_index].reshape(28, 28)  

# Predict the value using the trained model
test_image_flat = X_test[test_index].reshape(1, -1) 
y_pred = h(test_image_flat, W, b)
predicted_value = np.argmax(h(test_image_flat, W, b))

# True value 
true_value = test_data['label'].values[test_index]

# Display the model's guess and the true value
print("Probability Distribution:", y_pred)
print(f"Model's guess: {predicted_value}")
print(f"True value: {true_value}")
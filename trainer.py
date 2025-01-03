# The purpose of this file is to cleanly implement a feedforward neural network trained on the MNIST dataset ("Hello World" of neural nets) 

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 

path = os.listdir("../feedforward net/input")
print(path)

training_data = pd.read_csv("train.csv")
testing_data = pd.read_csv("test.csv")

# Notice the shape of our data, we have 60000 training images with 784 columns (excluding 'label', 784 pixels, 28x28 images)
print(training_data[:10]) # print the first 10 images
print(training_data.shape)

# Store labels 
y_train = training_data['label'].values

# Store pixels 
x_train = training_data.drop(columns='label').values / 255 # We divide by 255 to set throw everything in a range of 0 to 1  







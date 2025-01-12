import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the test (or training) data
test_data = pd.read_csv('./datasets/test.csv')

# Ensure correct number of features
X_test = test_data.drop(columns=['label']).values / 255  

# You can change this index to select a different test image
test_index = 5  

# Reshape the image to 28x28
test_image = X_test[test_index].reshape(28, 28)

# plot the sample 
fig = plt.figure
plt.imshow(test_image, cmap='gray')
plt.show()

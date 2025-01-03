import numpy as np 
import pandas as pd 



test_data = pd.read_csv("test.csv")

print(test_data.shape)
X_test = test_data.drop(columns=['label']).values / 255  # Ensure correct number of features

print(X_test.shape)


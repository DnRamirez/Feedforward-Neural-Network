# Matrix multiplication is a fundamental operation when working with neural networks 
# A matrix is a 2D array of numbers consisting of rows and columns
# A matrix with 2 rows and 3 columns is a 2x3 matrix
# GPUs are best suited for matrix multiplication because they can perform many calculations in parallel
import numpy as np 
op1 = np.array([[1, 2, 3],
                [1, 4, 5]])

print(f"Shape of first matrix is: {op1.shape}")

op2 = np.array([[1],
                [2],
                [3]])

print(f"Shape of second matrix is: {op2.shape}") 

# You can multiply these matrices because the inner dimensions match (3 == 3)
dotproduct = np.matmul(op1, op2)
print("product shape is: ", dotproduct.shape)
print("product result is: ", dotproduct)










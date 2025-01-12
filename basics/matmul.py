import numpy as np 

op1 = np.array([[1, 2, 3],
                [1, 4, 5]])

print(op1.shape)

op2 = np.array([[1],
                [2],
                [3]])

print(op2.shape) # should output 3,1 

# now you can multiply these because the inner dimensions match
dotproduct = np.matmul(op1, op2)
print(dotproduct)

op3 = np.array([[1, 2],
                [4, 5],
                [7, 8]])

print(op3.shape)

op4 = np.array([[3],
                [4]])

print(op4.shape)

product = np.matmul(op3, op4)
print("product shape is: ", product.shape)
print("product result is: ", product)










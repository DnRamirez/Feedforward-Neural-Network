# reshaping an array, there are some rules you need to follow 
# purpose: Matrix Multiplication: Many mathematical operations, such as matrix multiplication, require specific shapes. Reshaping ensures that the dimensions of arrays align properly for operations like dot products or other linear algebra computations.
# Broadcasting: Reshaping allows arrays to be broadcast together, enabling arithmetic operations between arrays of different shapes.
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
print("Before reshape: ", dotproduct)

# reshape ops 

# case 1, stays same 
z = dotproduct.reshape(-1, 1)
print("after reshape: ", z.shape)

#case 2: flatten 
z_flat = dotproduct.reshape(-1)  # This will change the shape to (2,)
print("Flattened shape: ", z_flat.shape)

# rules for reshape : If you have an array of shape (a, b), the total number of elements is a * b.
# When reshaping, ensure that the new shape's dimensions multiply to the same total: c * d = a * b.
# For example, a (4, 3) array has 12 elements. You can reshape it to (6, 2) or (3, 4) since both have 12 elements.






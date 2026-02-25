import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[5, 6],
              [7, 8]])

addition = A + B

matrix_multiplication = A @ B

element_multiplication = A * B

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Addition:\n", addition)
print("Matrix Multiplication:\n", matrix_multiplication)
print("Element Multiplication:\n", element_multiplication)
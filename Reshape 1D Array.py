import numpy as np

arr = np.arange(12)

matrix_3x4 = arr.reshape(3, 4)
matrix_2x6 = arr.reshape(2, 6)

print("Original Array:", arr)
print("Reshaped to 3x4:\n", matrix_3x4)
print("Reshaped to 2x6:\n", matrix_2x6)
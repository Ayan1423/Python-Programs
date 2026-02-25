import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60])

# Indexing
print("First element:", arr[0])
print("Last element:", arr[-1])

# Slicing
print("Elements from index 1 to 4:", arr[1:5])
print("Every second element:", arr[::2])
print("Reverse array:", arr[::-1])
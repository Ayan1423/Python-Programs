import pandas as pd

# Sample DataFrame
data = {
    'A': [10, 20, 30, 40, 50],
    'B': [5, 15, 25, 35, 45],
    'C': [2, 4, 6, 8, 10],
    'D': [1, 3, 5, 7, 9]
}

df = pd.DataFrame(data)
print("DataFrame:\n", df)

# (i) Average of second column (B)
avg_second_col = df.iloc[:, 1].mean()
print("\nAverage of second column:", avg_second_col)

# (ii) Average of first 5 rows of 3rd and 4th columns (C and D)
avg_c_d = df.iloc[:5, 2:4].mean()
print("\nAverage of C and D columns:\n", avg_c_d)

# (iii) Row-wise sum
row_sum = df.sum(axis=1)
print("\nRow-wise sum:\n", row_sum)

# (iv) Maximum of row-wise average
row_avg = df.mean(axis=1)
max_avg = row_avg.max()
print("\nMaximum of row-wise averages:", max_avg)
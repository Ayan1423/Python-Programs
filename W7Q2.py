import pandas as pd

# Load CSV file
df = pd.read_csv("data.csv")

# Display first 5 rows
print("Head:\n", df.head())

# Display last 5 rows
print("\nTail:\n", df.tail())

# Display dataset info
print("\nInfo:")
df.info()
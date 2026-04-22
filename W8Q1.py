import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())

# Check missing values (count per column)
print(df.isnull().sum())

# Percentage of missing values
print((df.isnull().sum() / len(df)) * 100)

print(df.info())
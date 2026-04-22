import pandas as pd

df = pd.read_csv("data.csv")

df['Age'] = df['Age'].fillna(df['Age'].mean())

df['Salary'] = df['Salary'].fillna(df['Salary'].median())

df['City'] = df['City'].fillna(df['City'].mode()[0])

df = df.dropna()

print(df.head())
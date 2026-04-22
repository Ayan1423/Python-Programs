import pandas as pd

df = pd.read_csv("data.csv")

df_clean = df.dropna()

df_filtered = df[df['Age'] >= 50]
print(df_filtered)

df_filtered = df[(df['Age'] > 30) & (df['Salary'] > 50000)]
print(df_filtered)

df_subset = df[['Age', 'Salary']]
print(df_subset)

df = df[df['Age'].notnull()]
print(df.head())
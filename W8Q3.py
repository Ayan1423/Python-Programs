import pandas as pd

df = pd.read_csv("data.csv")

df.rename(columns={
    'Name': 'new_name1',
    'Marks': 'new_name2'
}, inplace=True)

df['Age'] = df['Age'].astype(int)
df['Salary'] = df['Salary'].astype(float)
print(df.head())
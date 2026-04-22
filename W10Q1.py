import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sample1.csv")

print(df.head())

# ----- Line Plot -----
plt.figure(figsize=(8,5))
plt.plot(df['Column1'], df['Column2'], marker='o')
plt.title("Line Plot")
plt.xlabel("Column1")
plt.ylabel("Column2")
plt.grid(True)
plt.show()

# ----- Bar Plot -----
plt.figure(figsize=(8,5))
plt.bar(df['Category'], df['Values'], color='skyblue')
plt.title("Bar Plot")
plt.xlabel("Category")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("sample1.csv")

# ----- Histogram -----
plt.figure(figsize=(8,5))
sns.histplot(df['Column2'], kde=True, bins=20)
plt.title("Histogram")
plt.show()

# ----- Boxplot -----
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Column2'])
plt.title("Boxplot")
plt.show()
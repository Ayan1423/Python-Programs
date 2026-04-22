import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("sample1.csv")

# ----- Scatter Plot -----
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Column1'], y=df['Column2'])
plt.title("Scatter Plot")
plt.show()

# ----- Correlation Heatmap -----
plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()
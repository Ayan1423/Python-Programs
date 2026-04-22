import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sample.csv")

# Select only numerical columns
num_df = df.select_dtypes(include=[np.number])

# Correlation matrix
corr_matrix = num_df.corr()

# Plot heatmap
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Remove highly correlated features (threshold = 0.9)
threshold = 0.9
corr_pairs = corr_matrix.abs()

upper_triangle = corr_pairs.where(
    np.triu(np.ones(corr_pairs.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

df_reduced = df.drop(columns=to_drop)

print("Dropped columns:", to_drop)
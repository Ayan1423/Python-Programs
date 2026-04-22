import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("sample.csv")

# Select only numerical columns
num_df = df.select_dtypes(include=[np.number])

# Boxplots for each numerical column
for col in num_df.columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()

# Function to remove outliers using IQR
def remove_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return data[(data[column] >= lower) & (data[column] <= upper)]

# Apply to all numerical columns
df_clean = df.copy()
for col in num_df.columns:
    df_clean = remove_outliers(df_clean, col)

print("Shape after removing outliers:", df_clean.shape)
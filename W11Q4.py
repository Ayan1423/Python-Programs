# Problem 4

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model 1: Linear Regression

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

# Model 2: Decision Tree Regression

dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

dt_mae = mean_absolute_error(y_test, dt_pred)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))

# Comparison

print("Model Comparison:\n")

print("Linear Regression:")
print("MAE:", lr_mae)
print("RMSE:", lr_rmse)

print("\nDecision Tree Regression:")
print("MAE:", dt_mae)
print("RMSE:", dt_rmse)

# Visualisation
import matplotlib.pyplot as plt

plt.figure(figsize=(12,5))

# Linear Regression plot
plt.subplot(1,2,1)
plt.scatter(y_test, lr_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Linear Regression")

# Decision Tree plot
plt.subplot(1,2,2)
plt.scatter(y_test, dt_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.title("Decision Tree Regression")

plt.show()
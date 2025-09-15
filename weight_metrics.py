# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv('exercise_dataset.csv')
print("Initial Data Preview:")
print(df.head())

# Drop ID column as it's not needed
df.drop('ID', axis=1, inplace=True)

# Show basic info
print("\nDataset Info:")
print(df.info())

# Statistical summary
print("\nDescriptive Statistics:")
print(df.describe())

# Unique values in 'Exercise' column
print("\nExercise Types:")
print(df['Exercise'].unique())

# Feature selection and target variable
x = df.drop('Calories', axis=1)
y = df['Calories']

# Splitting dataset into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# --- LINEAR REGRESSION MODEL ---
reg = LinearRegression()
model = reg.fit(x_train, y_train)
print("\nLinear Regression R² Score on Test Set:", model.score(x_test, y_test))

# --- K-NEAREST NEIGHBORS MODEL ---
knn = KNeighborsRegressor(n_neighbors=8)
model2 = knn.fit(x_train, y_train)
print("KNN R² Score on Test Set:", model2.score(x_test, y_test))

# --- CROSS VALIDATION ---
print("\nCross-Validation (Linear Regression)")
print("R² Score (CV):", cross_val_score(model, x_train, y_train, cv=10, scoring='r2').mean())
print("RMSE (CV):", np.sqrt(-cross_val_score(model, x_train, y_train, cv=10, scoring='neg_mean_squared_error')).mean())

print("\nCross-Validation (KNN)")
print("R² Score (CV):", cross_val_score(model2, x_train, y_train, cv=10, scoring='r2').mean())
print("RMSE (CV):", np.sqrt(-cross_val_score(model2, x_train, y_train, cv=10, scoring='neg_mean_squared_error')).mean())

# --- EVALUATION METRICS FOR LINEAR REGRESSION ---
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nLinear Regression Evaluation Metrics:")
print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# --- EVALUATION METRICS FOR KNN ---
y_pred_knn = model2.predict(x_test)
mae_knn = mean_absolute_error(y_test, y_pred_knn)
mse_knn = mean_squared_error(y_test, y_pred_knn)
rmse_knn = np.sqrt(mse_knn)
r2_knn = r2_score(y_test, y_pred_knn)

print("\nKNN Evaluation Metrics:")
print(f"MAE: {mae_knn}")
print(f"MSE: {mse_knn}")
print(f"RMSE: {rmse_knn}")
print(f"R²: {r2_knn}")
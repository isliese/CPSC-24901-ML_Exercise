# Supervised learning for linear regression - least_sq.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load data from CSV file
def load_data(file_name):
    data = pd.read_csv(file_name)
    X = data.iloc[:, 0].values  # Assuming first column is independent variable
    y = data.iloc[:, 1].values  # Assuming second column is dependent variable
    return X, y

# Compute the least squares estimates
def least_squares(X, y):
    n = len(X)
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    b1 = numerator / denominator
    b0 = y_mean - b1 * X_mean
    
    return b0, b1

# Predict function
def predict(X, b0, b1):
    return b0 + b1 * X

# Plot results
def plot_regression(X, y, b0, b1):
    plt.scatter(X, y, color='blue', label='Data points')  
    plt.plot(X, predict(X, b0, b1), color='red', linewidth=1, label='Least Squares Fit')  # 회귀선 (얇게 설정)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__) 
    file_name = os.path.join(base_dir, "Q2.csv")

    X, y = load_data(file_name)
    b0, b1 = least_squares(X, y)
    
    print(f"Intercept (b0): {b0}")
    print(f"Slope (b1): {b1}")
    
    plot_regression(X, y, b0, b1)

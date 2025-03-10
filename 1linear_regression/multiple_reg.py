import os
import numpy as np
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

script_dir = os.path.dirname(os.path.abspath(__file__)) 
file_path = os.path.join(script_dir, "Q2.csv")  # Relative path

def csv_file_loader(file_name):
    with open(file_name, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        data = list(csvreader)
    return data

data = csv_file_loader(file_path)

# Convert data to NumPy array and separate X and y
data = np.array(data[1:], dtype=float)  # Assuming first row is header
X = data[:, :-1]  # X values excluding the last column
y = data[:, -1]   # The last column as y values

# Train multiple linear regression model
model = LinearRegression()
model.fit(X, y)

# Print coefficients and intercept
coefficients = [model.intercept_] + list(model.coef_)
print(f"Regression coefficients (w0, w1, ..., wn): {coefficients}")

# Select a random X value and make a prediction
random_index = np.random.randint(0, len(X))
x_sample = X[random_index].reshape(1, -1)
y_pred = model.predict(x_sample)
print(f"Random x value: {x_sample.flatten()}")
print(f"Predicted y value: {y_pred[0]}")

# Compute RMSE
y_pred_all = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred_all))
print(f"RMSE: {rmse}")

# Evaluate model performance based on RMSE
if rmse < 10:
    print("This function has high prediction accuracy.")
else:
    print("This function has low prediction accuracy and needs improvement.")

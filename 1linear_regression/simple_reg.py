# Supervised learning for linear regression - simple_reg.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("The dots are the actual observed data values")
print(" The line means the straight line ** found to fit the data as well as possible after the model has learned the patterns of the data.")

# First dataset
# Reshape is used to convert 1D array into a 2D array
# because sklearn's LinearRegression requires input as (samples, features)
X1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
Y1 = np.array([2.5, 4.1, 5.6, 7.2, 8.8, 10.3, 11.9, 13.5, 15.0, 16.8])

# Create linear regression model and learn
model1 = LinearRegression()
model1.fit(X1, Y1)

# Print Regression 
y_pred_100 = model1.predict(np.array([[100]]))
print(f"\nThe first dataset regression: y = {model1.intercept_:.2f} + {model1.coef_[0]:.2f}x")
print(f"Predicted value when x = 100: {y_pred_100[0]:.2f}")

# Visualize the first dataset 
fig1 = plt.figure()  
fig1.canvas.manager.set_window_title("First Dataset Visualization")  
plt.scatter(X1, Y1, color='blue', label='Actual Data')
plt.plot(X1, model1.predict(X1), color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression for First Dataset')
plt.legend()
plt.show()


# Second dataset
# Reshape is used to convert 1D array into a 2D array
# because sklearn's LinearRegression requires input as (samples, features)
X2 = np.array([-3, -2.5, -2, -1.5, -1, 0, 1, 1.5, 2, 2.5, 3]).reshape(-1, 1)
Y2 = np.array([17.5, 12.9, 9.5, 7.2, 5.8, 5.5, 7.1, 9.7, 13.5, 18.4, 24.4])

# Create linear regression model and learn
model2 = LinearRegression()
model2.fit(X2, Y2)

# Print Regression
y_pred_05 = model2.predict(np.array([[0.5]]))
print(f"Second dataset regression: y = {model2.intercept_:.2f} + {model2.coef_[0]:.2f}x")
print(f"Predicted value when x = 0.5: {y_pred_05[0]:.2f}\n")

# Visualize second dataset
fig2 = plt.figure()  
fig2.canvas.manager.set_window_title("Second Dataset Visualization") 
plt.scatter(X2, Y2, color='green', label='Actual Data')
plt.plot(X2, model2.predict(X2), color='orange', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression for Second Dataset')
plt.legend()
plt.show()


# Error Calculation for first dataset

# Calculate expected error
y_pred1 = model1.predict(X1)
errors1 = Y1 - y_pred1  # Difference between real value and expected value
total_error1 = np.sum(errors1)  # Total Error: ∑(yi - ŷi)
squared_errors1 = errors1 ** 2  # Squared Error: (yi - ŷi)²
sum_squared_error1 = np.sum(squared_errors1)  # Sum of Squared Error: ∑(yi - ŷi)²
mean_squared_error1 = np.mean(squared_errors1)  # Mean Squared Error(MSE): (1/N)∑(yi - ŷi)²
root_mean_squared_error1 = np.sqrt(mean_squared_error1)  # Root Mean Squared Error(RMSE): √(1/N)∑(yi - ŷi)²

print("\nFirst dataset error calculations:")
print(f"Total Error: {total_error1:.2f}")  
print(f"Sum of Squared Errors (SSE): {sum_squared_error1:.2f}")  
print(f"Mean Squared Error (MSE): {mean_squared_error1:.2f}")  
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error1:.2f}")  

# Error Calculation for second dataset

# Calculate expected error
y_pred2 = model2.predict(X2)
errors2 = Y2 - y_pred2  # Difference between real value and expected value
total_error2 = np.sum(errors2) # Total Error: ∑(yi - ŷi)
squared_errors2 = errors2 ** 2  # Squared Error: (yi - ŷi)²
sum_squared_error2 = np.sum(squared_errors2)  # Sum of Squared Error: ∑(yi - ŷi)²
mean_squared_error2 = np.mean(squared_errors2)  # Mean Squared Error(MSE): (1/N)∑(yi - ŷi)²
root_mean_squared_error2 = np.sqrt(mean_squared_error2)  # Root Mean Squared Error(RMSE): √(1/N)∑(yi - ŷi)²

print("\nSecond dataset error calculations:")
print(f"Total Error: {total_error2:.2f}")
print(f"Sum of Squared Errors (SSE): {sum_squared_error2:.2f}") 
print(f"Mean Squared Error (MSE): {mean_squared_error2:.2f}")  
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error2:.2f}") 

# Comparison and Explanation of Error Measures

print("\nComparison of error measures:")
print("1. Total Error is simply the sum of the differences between actual and predicted values. "
      "However, this measure can be misleading because it does not account for the magnitude of errors or the number of data points.")
print("2. Sum of Squared Errors (SSE) increases significantly as errors grow, making it useful for assessing how well the model fits the data. "
      "However, it does not consider the number of data points or the scale of the data.")
print("3. Mean Squared Error (MSE) calculates the average of squared errors, taking the number of data points into account. "
      "It is a useful metric for comparing models.")
print("4. Root Mean Squared Error (RMSE) is the square root of MSE, which converts the error back to the original unit of the data, making it more intuitive.")

# MSE and RMSE are more useful for comparing model performance regardless of data size or the number of points.
# RMSE provides an error measure in the same unit as the data, making it easier to interpret the actual magnitude of errors.
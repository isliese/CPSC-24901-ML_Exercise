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
# 첫 번째 데이터셋에 대한 오류 계산

# 예측값을 계산
y_pred1 = model1.predict(X1)
errors1 = Y1 - y_pred1  # 실제 값과 예측 값의 차이
total_error1 = np.sum(errors1)  # 총 오류: ∑(yi - ŷi)
squared_errors1 = errors1 ** 2  # 제곱 오류: (yi - ŷi)²
sum_squared_error1 = np.sum(squared_errors1)  # 제곱 오류의 합: ∑(yi - ŷi)²
mean_squared_error1 = np.mean(squared_errors1)  # 평균 제곱 오류(MSE): (1/N)∑(yi - ŷi)²
root_mean_squared_error1 = np.sqrt(mean_squared_error1)  # 평균 제곱근 오류(RMSE): √(1/N)∑(yi - ŷi)²

print("\nFirst dataset error calculations:")
print(f"Total Error: {total_error1:.2f}")  # 총 오류 출력
print(f"Sum of Squared Errors (SSE): {sum_squared_error1:.2f}")  # 제곱 오류 합 출력
print(f"Mean Squared Error (MSE): {mean_squared_error1:.2f}")  # 평균 제곱 오류 출력
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error1:.2f}")  # 제곱근 평균 제곱 오류 출력

# Error Calculation for second dataset
# 두 번째 데이터셋에 대한 오류 계산

# 예측값을 계산
y_pred2 = model2.predict(X2)
errors2 = Y2 - y_pred2  # 실제 값과 예측 값의 차이
total_error2 = np.sum(errors2)  # 총 오류: ∑(yi - ŷi)
squared_errors2 = errors2 ** 2  # 제곱 오류: (yi - ŷi)²
sum_squared_error2 = np.sum(squared_errors2)  # 제곱 오류의 합: ∑(yi - ŷi)²
mean_squared_error2 = np.mean(squared_errors2)  # 평균 제곱 오류(MSE): (1/N)∑(yi - ŷi)²
root_mean_squared_error2 = np.sqrt(mean_squared_error2)  # 평균 제곱근 오류(RMSE): √(1/N)∑(yi - ŷi)²

print("\nSecond dataset error calculations:")
print(f"Total Error: {total_error2:.2f}")  # 총 오류 출력
print(f"Sum of Squared Errors (SSE): {sum_squared_error2:.2f}")  # 제곱 오류 합 출력
print(f"Mean Squared Error (MSE): {mean_squared_error2:.2f}")  # 평균 제곱 오류 출력
print(f"Root Mean Squared Error (RMSE): {root_mean_squared_error2:.2f}")  # 제곱근 평균 제곱 오류 출력

# Comparison and Explanation of Error Measures
# 오류 측정값의 비교와 설명

print("\nComparison of error measures:")
print("1. Total Error는 실제 값과 예측 값의 차이를 단순히 더한 값입니다. "
      "하지만 이 값은 오류의 크기나 데이터 포인트 수를 고려하지 않기 때문에 다소 misleading할 수 있습니다.")
print("2. Sum of Squared Errors (SSE)는 오류가 클수록 더 큰 값을 가지게 되어, 모델이 데이터에 얼마나 잘 맞는지 확인하는 데 유용합니다. "
      "하지만 데이터 포인트의 수나 데이터의 크기를 고려하지 않습니다.")
print("3. Mean Squared Error (MSE)는 제곱 오류의 평균을 내므로, 데이터 포인트 수에 대한 영향을 고려할 수 있습니다. "
      "모델을 비교할 때 유용한 척도가 됩니다.")
print("4. Root Mean Squared Error (RMSE)는 MSE의 제곱근으로, 오류 값을 원래 데이터의 단위로 되돌려주기 때문에 더 직관적입니다.")

# MSE와 RMSE는 데이터의 크기나 포인트 수에 관계없이 모델의 성능을 비교할 때 더 유용한 오류 측정값입니다.
# RMSE는 해석하기 쉬운 단위를 제공하여, 실제 오류의 크기를 더 명확히 알 수 있게 해줍니다.


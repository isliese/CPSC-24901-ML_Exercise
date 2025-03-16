#! usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()
X = digits.data  # Feature vectors (8x8 pixel images flattened into 64 features)
y = digits.target  # Labels (digits 0-9)

# Split into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize feature values (important for MLP and K-NN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the MLP Classifier (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                    learning_rate_init=0.001, max_iter=500, batch_size=32, momentum=0.9, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Predict with MLP
y_pred_mlp = mlp.predict(X_test_scaled)

# Train K-NN Classifier for comparison
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn.fit(X_train_scaled, y_train)

# Predict with K-NN
y_pred_knn = knn.predict(X_test_scaled)

# Compute accuracy
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

# Print results
print(f"MLP Accuracy: {accuracy_mlp:.4f}")
print(f"K-NN Accuracy: {accuracy_knn:.4f}")

# Tune MLP: Testing different hidden layer sizes
hidden_layer_sizes = [(50,), (100,), (100, 50), (200, 100)]
mlp_accuracies = []

for size in hidden_layer_sizes:
    mlp_tuned = MLPClassifier(hidden_layer_sizes=size, activation='relu', solver='adam',
                              learning_rate_init=0.001, max_iter=500, batch_size=32, momentum=0.9, random_state=42)
    mlp_tuned.fit(X_train_scaled, y_train)
    y_pred_tuned = mlp_tuned.predict(X_test_scaled)
    mlp_accuracies.append(accuracy_score(y_test, y_pred_tuned))

# Plot accuracy for different hidden layer sizes
plt.figure(figsize=(8, 5))
plt.plot([str(size) for size in hidden_layer_sizes], mlp_accuracies, marker='o', linestyle='dashed', color='blue', label="MLP Accuracy")
plt.axhline(y=accuracy_knn, color='red', linestyle='--', label="K-NN Accuracy")
plt.xlabel("Hidden Layer Sizes")
plt.ylabel("Accuracy")
plt.title("MLP Accuracy vs. Hidden Layer Configuration (Compared to K-NN)")
plt.legend()
plt.show()


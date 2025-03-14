import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the digits dataset
digits = load_digits()
X = digits.data  # Features (flattened 8x8 images)
y = digits.target  # Labels (digits 0-9)

# Display the first image from the dataset
plt.figure(figsize=(2, 2))
plt.imshow(digits.images[0], cmap='gray', interpolation='none')
plt.title(f"Label: {digits.target[0]}")
plt.axis('off')
plt.show()

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate K-NN with different parameters
def evaluate_knn(k_values, weights, X_train, X_test, y_train, y_test):
    results = []
    for k in k_values:
        for weight in weights:
            knn = KNeighborsClassifier(n_neighbors=k, weights=weight)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results.append((k, weight, accuracy))
            print(f"K-NN Accuracy (k={k}, weights='{weight}'): {accuracy * 100:.2f}%")
    return results

# Test different k values and weighting schemes
k_values = [3, 5, 7, 9]  # Test different numbers of neighbors
weights = ['uniform', 'distance']  # Test uniform and distance-based weighting
results = evaluate_knn(k_values, weights, X_train, X_test, y_train, y_test)

# Find the best parameters
best_k, best_weight, best_accuracy = max(results, key=lambda x: x[2])
print(f"\nBest parameters: k={best_k}, weights='{best_weight}' with accuracy: {best_accuracy * 100:.2f}%")

# Plot the results for visualization
k_results = {}
for k in k_values:
    k_results[k] = [result[2] for result in results if result[0] == k]

plt.figure(figsize=(10, 6))
for k, accuracies in k_results.items():
    plt.plot(weights, accuracies, marker='o', label=f'k={k}')
plt.title("K-NN Accuracy for Different Parameters")
plt.xlabel("Weighting Scheme")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()



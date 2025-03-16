#! usr/bin.python3

# kmeans.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data  # Features
    y = iris.target  # Labels
    return X, y

# K-means clustering
def kmeans_clustering(X, K=3):
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)
    return kmeans

# RMSE Calculation for each cluster
def compute_rmse(X, kmeans):
    clusters = kmeans.predict(X)
    centroids = kmeans.cluster_centers_
    
    rmse_per_cluster = []
    for i in range(kmeans.n_clusters):
        cluster_points = X[clusters == i]
        centroid = centroids[i]
        mse = mean_squared_error(cluster_points, np.tile(centroid, (cluster_points.shape[0], 1)))
        rmse = np.sqrt(mse)
        rmse_per_cluster.append(rmse)
    
    return rmse_per_cluster

# KNN classifier for evaluation
def knn_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    return accuracy

# Main program
def main():
    X, y = load_iris_data()
    
    # Apply K-means clustering
    kmeans = kmeans_clustering(X, K=3)
    
    # Compute RMSE for each cluster
    rmse_per_cluster = compute_rmse(X, kmeans)
    print(f"RMSE per cluster: {rmse_per_cluster}")
    
    # Apply KNN classification for comparison
    accuracy = knn_classifier(X, y)
    print(f"KNN classification accuracy: {accuracy * 100:.2f}%")
    
    # Compare cluster alignment with actual classes
    print("\nCluster Label vs Actual Class Label:")
    clusters = kmeans.predict(X)
    for i in range(3):  # There are 3 clusters
        cluster_class = np.bincount(y[clusters == i]).argmax()  # Find the majority class in the cluster
        print(f"Cluster {i} most frequent class: {cluster_class}")
    
if __name__ == "__main__":
    main()

#Here's a simple implementation of the K-Means Clustering Algorithm in Python using numpy and matplotlib for visualization:

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def fit(self, X):
        # Initialize centroids randomly from the dataset
        np.random.seed(42)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for _ in range(self.max_iters):
            # Assign clusters based on the closest centroid
            clusters = self._assign_clusters(X)

            # Compute new centroids
            new_centroids = np.array([
                X[clusters == i].mean(axis=0) if np.any(clusters == i) else self.centroids[i] 
                for i in range(self.k)
            ])

            # Check for convergence (if centroids don't change)
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return self.centroids

    def _assign_clusters(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X):
        return self._assign_clusters(X)

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Apply K-Means clustering
kmeans = KMeans(k=3)
centroids = kmeans.fit(X)
clusters = kmeans.predict(X)

# Plot results
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
'''
Explanation:

1. Initialization: Randomly selects k centroids from the dataset.


2. Cluster Assignment: Assigns each data point to the closest centroid.


3. Centroid Update: Computes new centroids based on the mean of assigned points.


4. Convergence Check: Stops if centroids do not change between iterations.


5. Visualization: Uses matplotlib to plot the clustered data and centroids.



This is a basic version without optimizations like K-Means++. Let me know if you want improvements!
'''

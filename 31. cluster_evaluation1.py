from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Step 1: Generate a toy dataset
# TODO: Decide number of samples and centers
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Step 2: Try different values of k
for k in [2, 3, 4, 5]:   # TODO: Try different values for k
    # Train KMeans with k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)   # TODO: Fill in n_clusters
    labels = kmeans.fit_predict(X)
    
    # Step 3: Evaluate clustering
    inertia = kmeans.inertia_    # TODO: Access inertia value
    score = silhouette_score(X, labels)   # TODO: Pass correct data and labels
    
    print(f"k = {k}, Inertia = {inertia}, Silhouette Score = {score}")

# Step 4: Plot the best clustering
best_k = 2   # TODO: Select the best k based on scores
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X)

# the clusters and centroids
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label="Centroids")
plt.title(f"K-Means Clustering with {best_k} Clusters")
plt.legend()
plt.savefig("clusters_eval_practice.png")
plt.close()

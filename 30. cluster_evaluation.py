from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate toy dataset with 3 clusters
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)

# Train KMeans with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)

print("Inertia (lower is better):", kmeans.inertia_)

score = silhouette_score(X, labels)
print("Silhouette Score (closer to 1 is better):", score)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, label="Centroids")
plt.title("K-Means Clustering with 3 Clusters")
plt.legend()
plt.savefig("clusters_eval.png")
plt.close()
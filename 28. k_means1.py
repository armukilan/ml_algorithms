# TODO: Import required libraries
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("customer_data.csv")

# Step 2: Select features
X = df[[ "CustomerID","Age" ]].values

# Step 3: Initialize KMeans
kmeans = KMeans(n_clusters=2, random_state=42)

# Step 4: Fit the model and get labels
labels = kmeans.fit_predict(X)

# Step 5: Visualize the clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label="Centroids")
plt.legend()
plt.savefig("customer_clusters.png")
plt.close()

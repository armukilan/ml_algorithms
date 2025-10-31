from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data    # features (4 dimensions: sepal length, sepal width, petal length, petal width)
y = iris.target  # labels (species)

# Step 2: Apply PCA to reduce 4D data → 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Step 3: Visualize the PCA results
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA on Iris Dataset")

# Save and show the plot
plt.savefig("pca_iris.png")
plt.show()
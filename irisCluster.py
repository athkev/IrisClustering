import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

#load data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
data = pd.read_csv(url, header=None, names=names)

X = data.drop('class', axis=1)

# Preprocessing: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Define number of clusters
k = 3

kmeans = KMeans(n_clusters=k, random_state=0)
kmeans.fit(X)

cluster_labels = kmeans.labels_

# Add cluster labels to data frame
data["cluster"] = cluster_labels

# Extract cluster-wise data
cluster_data = {}
for i in range(k):
    cluster_data[i] = data[data["cluster"] == i]

# Sepal Length vs. Width
plt.figure(figsize=(8, 6))
for cluster_id, cluster in cluster_data.items():
    plt.scatter(cluster["sepal_length"], cluster["sepal_width"], label=f"Cluster {cluster_id}")
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris K-Means Clustering - Sepal Length vs. Width")
plt.show()
plt.close()

# Petal Length vs. Width
plt.figure(figsize=(8, 6))
for cluster_id, cluster in cluster_data.items():
    plt.scatter(cluster["petal_length"], cluster["petal_width"], label=f"Cluster {cluster_id}")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Iris K-Means Clustering - Petal Length vs. Width")
plt.show()

# Calculate silhouette scores
sepal_silhouette = silhouette_score(X[["sepal_length", "sepal_width"]], cluster_labels)
petal_silhouette = silhouette_score(X[["petal_length", "petal_width"]], cluster_labels)

# Print results
print("Sepal Silhouette Score:", sepal_silhouette)
print("Petal Silhouette Score:", petal_silhouette)
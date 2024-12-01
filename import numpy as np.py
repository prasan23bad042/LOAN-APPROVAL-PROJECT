import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(X)

plt.scatter(X[cluster_labels == 0, 0], X[cluster_labels == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[cluster_labels == 1, 0], X[cluster_labels == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[cluster_labels == 2, 0], X[cluster_labels == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[cluster_labels == 3, 0], X[cluster_labels == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[cluster_labels == 4, 0], X[cluster_labels == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.title('Agglomerative Clustering')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
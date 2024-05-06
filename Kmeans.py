import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score


data = pd.read_csv('C:\\Users\\abhin\\Downloads\\Seed_Data.csv')
print(data.head())

X = data[['A', 'P']].values


n_clusters = 3 
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
centers = kmeans.cluster_centers_

calinski_harabasz_score = calinski_harabasz_score(X, y_kmeans)
print("Calinski-Harabasz Index:", calinski_harabasz_score)


plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title('KMeans Clustering')
plt.xlabel('A')
plt.ylabel('P')
plt.show()
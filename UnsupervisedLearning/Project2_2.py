import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate 2 random centroids
mean1 = [np.random.randint(100), np.random.randint(100)]
mean2 = [np.random.randint(100), np.random.randint(100)]

# Generate 2 random clusters 100 random points in a 100 by 100 grid
cov = [[100, 0], [0, 100]]
x1, y1 = np.random.multivariate_normal(mean1, cov, 100).T
x2, y2 = np.random.multivariate_normal(mean2, cov, 100).T
x = np.append(x1, x2)
y = np.append(y1, y2)
X = []
for a in range(0, x.__len__()):
    tmp = [x[a], y[a]]
    X.append(tmp)

# Plot the data, most likely be 2 distinct clusters
plt.plot(x, y, '.')
plt.axis('equal')

# Creating the KMC model
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_
print(labels)

colors = ["g.", "r."]

for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:, 0], centroids[:, 1], marker = "X", s = 150, zorder = 10)

plt.show()

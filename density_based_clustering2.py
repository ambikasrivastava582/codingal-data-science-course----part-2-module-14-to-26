import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("customers.csv", index_col=0)

df.head()

# Select Annual Income and Spending Score
x = df[['Annual Income (k$)', 'Spending Score (1-100)']].values

x.shape


# -----------------------------
# DBSCAN Clustering
# -----------------------------

from sklearn.cluster import DBSCAN

db = DBSCAN(
    eps=3,
    min_samples=4,
    metric='euclidean'
)

model = db.fit(x)

label = model.labels_

print("Labels:")
print(label)


# Find number of clusters
from sklearn import metrics

sample_cores = np.zeros_like(label, dtype=bool)

sample_cores[db.core_sample_indices_] = True

n_clusters = len(set(label)) - (1 if -1 in label else 0)

print("No. of clusters:", n_clusters)


# Predict clusters
y_means = db.fit_predict(x)


# Display DBSCAN clusters
plt.figure(figsize=(7, 5))

plt.scatter(
    x[y_means == 0, 0],
    x[y_means == 0, 1],
    s=50,
    c='pink'
)

plt.scatter(
    x[y_means == 1, 0],
    x[y_means == 1, 1],
    s=50,
    c='yellow'
)

plt.scatter(
    x[y_means == 2, 0],
    x[y_means == 2, 1],
    s=50,
    c='cyan'
)

plt.scatter(
    x[y_means == 3, 0],
    x[y_means == 3, 1],
    s=50,
    c='magenta'
)

plt.scatter(
    x[y_means == 4, 0],
    x[y_means == 4, 1],
    s=50,
    c='orange'
)

plt.scatter(
    x[y_means == 5, 0],
    x[y_means == 5, 1],
    s=50,
    c='blue'
)

plt.scatter(
    x[y_means == 6, 0],
    x[y_means == 6, 1],
    s=50,
    c='red'
)

plt.scatter(
    x[y_means == 7, 0],
    x[y_means == 7, 1],
    s=50,
    c='black'
)

plt.scatter(
    x[y_means == 8, 0],
    x[y_means == 8, 1],
    s=50,
    c='violet'
)

plt.xlabel('Annual Income in (k)')
plt.ylabel('Spending Score from 1-100')
plt.title('Clusters of Data')

plt.show()


# -----------------------------
# Hierarchical Clustering
# -----------------------------

import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(
    sch.linkage(x, method='ward')
)

plt.title('Dendrogram', fontsize=20)
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')

plt.show()


# Create Hierarchical Clustering model
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(
    n_clusters=9,
    metric='euclidean',
    linkage='ward'
)

y_hc = hc.fit_predict(x)


# Display Hierarchical clusters

plt.scatter(
    x[y_hc == 0, 0],
    x[y_hc == 0, 1],
    s=50,
    c='pink'
)

plt.scatter(
    x[y_hc == 1, 0],
    x[y_hc == 1, 1],
    s=50,
    c='yellow'
)

plt.scatter(
    x[y_hc == 2, 0],
    x[y_hc == 2, 1],
    s=50,
    c='cyan'
)

plt.scatter(
    x[y_hc == 3, 0],
    x[y_hc == 3, 1],
    s=50,
    c='magenta'
)

plt.scatter(
    x[y_hc == 4, 0],
    x[y_hc == 4, 1],
    s=50,
    c='orange'
)

plt.scatter(
    x[y_hc == 5, 0],
    x[y_hc == 5, 1],
    s=50,
    c='blue'
)

plt.scatter(
    x[y_hc == 6, 0],
    x[y_hc == 6, 1],
    s=50,
    c='red'
)

plt.scatter(
    x[y_hc == 7, 0],
    x[y_hc == 7, 1],
    s=50,
    c='black'
)

plt.scatter(
    x[y_hc == 8, 0],
    x[y_hc == 8, 1],
    s=50,
    c='violet'
)

plt.title('Hierarchical Clustering', fontsize=20)

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')

plt.grid()

plt.show()
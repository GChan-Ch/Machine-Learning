from sklearn import datasets
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

iris = datasets.load_iris()

# Let's convert to dataframe
iris = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
 					columns=iris['feature_names'] + ['species'])

# Let's remove spaces from column name
iris.columns = iris.columns.str.replace(' ', '')
iris.head()

X = iris.iloc[:, :3] 	# independent variables
y = iris.species 	# dependent variables
sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

score = []
for n_clusters in range(2, 10):
 	kmeans = KMeans(n_clusters=n_clusters)
 	kmeans.fit(X)
 	labels = kmeans.labels_
 	centroids = kmeans.cluster_centers_
 	score.append(silhouette_score(X, labels, metric='euclidean'))

# Set the size of the plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(score)
plt.grid(True)
plt.ylabel("Silouette Score")
plt.xlabel("k")
plt.title("Silouette for K-means")

# Initialize the clusterer with n_clusters value and a random generator
model = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=0)
model.fit_predict(X)
cluster_labels = np.unique(model.labels_)
n_clusters = cluster_labels.shape[0]

# Compute the silhouette scores for each sample
silhouette_vals = silhouette_samples(X, model.labels_)

plt.subplot(1, 2, 2)
y_lower, y_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
 	c_silhouette_vals = silhouette_vals[cluster_labels ]
 	c_silhouette_vals.sort()
 	y_upper += len(c_silhouette_vals)
 	color = cm.nipy_spectral(float(i) / n_clusters)
 	plt.barh(range(y_lower, y_upper), c_silhouette_vals,
	 	facecolor=color, edgecolor=color, alpha=0.7)
 	yticks.append((y_lower + y_upper) / 2)
 	y_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)

plt.yticks(yticks, cluster_labels+1)

# The vertical line for average silhouette score of all the values
plt.axvline(x=silhouette_avg, color="red", linestyle="--")

plt.ylabel('Cluster')
plt.xlabel('Silhouette coefficient')
plt.title("Silouette for K-means")
plt.show()

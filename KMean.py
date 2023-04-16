#imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
import cv2

class KMeans():
	"""
	k-Means Clustering
	"""

	def __init__(self, n_clus):
		self.n_clus = n_clus	#Number of clusters
		self.centroids = None
		self.X = None
		self.clusters = None

	def getCentroids(self):
		""" Returns position of the centroids
		:return self.centroids: Centroids
		"""
		return self.centroids

	def getClusters(self):
		""" Returns clusters with shape (1, Npts).
			E.g. a data point X[i] belongs to
			cluster self.clusters[i].
		:return self.clusters: clusters
		"""
		return self.clusters

	def vizualize(self):
		""" Plot data points and final position of centroids.
			This function need to be called after training.
		"""
		plt.style.use("ggplot")
		plt.scatter(self.X[:,0], self.X[:,1], color = '#333333')
		Npts, Ndim = self.X.shape
		Nclus, _ = self.centroids.shape

		for k in range(Nclus):
			plt.scatter(self.centroids[k][0], self.centroids[k][1], marker = "*", color="#4169e1", s=130)

		plt.xlabel('x1')
		plt.ylabel('x2')
		plt.show()

	def fit(self, X, max_iter ,init_state = None):
		""" Train the K-Mean model, finds ultimate positions
		for centroids and create clusters.
		:param X: Data matrix with shape (Npts, Ndim)
		:param init_state: Initial state for centroids. If nothing is given,
		the model will generate random positions.
		:param max_iter: Maximum number of iterations
		"""
		Npts, Ndim = X.shape
		self.X = X

		if init_state is None:
			X_max, X_min = np.max(X), np.min(X)
			self.centroids = np.random.uniform(low = X_min, high=X_max, size = (self.n_clus,Ndim))
		else:
			self.centroids = init_state

		for i in range(max_iter):

			diff = cdist(X, self.centroids, metric="euclidean")
			self.clusters = np.argmin(diff, axis=1)

			for i in range(self.n_clus):

				self.centroids[i] = np.mean(X[np.where(self.clusters == i)] , axis=0)

	def predict(self, X):
		""" Predicts clusters for data points.
		:param X: Data point-s with shape (Npts, Ndim)
		:return: predicted clusters
		"""
		diff = cdist(X, self.centroids, metric="euclidean")
		return np.argmin(diff, axis=1)
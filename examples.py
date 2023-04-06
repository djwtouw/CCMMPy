import numpy as np
from ccmmpy import SparseWeights, CCMM


# %%

# Load data
X = np.genfromtxt("data/data.csv", delimiter=",")

# Generate a sparse weight matrix
W = SparseWeights(X, k=8, phi=3)

# Set a sequence for lambda
lambdas = np.arange(0, 350, 0.1)

# Compute the clusterpath given the lambdas
clust = CCMM(X, W).convex_clusterpath(lambdas)

# Plot the clusterpath and color the observations for the solution with four
# clusters
clust.plot_clusterpath(n_clusters=4)

# Change phi
W.phi = 4.5

# Compute the clusterpath given the lambdas
clust = CCMM(X, W).convex_clusterpath(lambdas)

# Plot the clusterpath and color the observations for the solution with four
# clusters
clust.plot_clusterpath(n_clusters=4)

# Get the cluster membership vector
labels = clust.clusters(4)

# Fails
labels = clust.clusters(93)

# %%

# Generate a data set
np.random.seed(12)
X = np.random.rand(10, 2)

# Generate a sparse weight matrix
W = SparseWeights(X, k=2, phi=3, connected=False)

# Set a sequence for lambda
lambdas = np.arange(0, 40, 0.01)

# Compute the clusterpath given the lambdas
clust = CCMM(X, W).convex_clusterpath(lambdas)

# Plot the clusterpath and color the observations for the solution with four
# clusters
clust.plot_clusterpath(draw_nz_weights=True)

# Scatter plot
clust.scatter(n_clusters=2, draw_nz_weights=True)

# %%

# Generate a data set
np.random.seed(2)
X = np.random.rand(10, 2)

# Generate a sparse weight matrix
W = SparseWeights(X, k=2, phi=3)

# Find two clusters
clust = CCMM(X, W).convex_clustering(target_low=2, target_high=2)

# Scatter plot of the result
clust.scatter(n_clusters=2)

# Find all clusterings in the range [1, 10]
clust = CCMM(X, W).convex_clustering(target_low=1, target_high=10)

# Plot the dendrogram
clust.plot_dendrogram()

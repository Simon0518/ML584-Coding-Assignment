import math
import pandas as pds
import numpy as np
from numpy import linalg as lag
import matplotlib.pyplot as plt
import sklearn.cluster as clu
import sklearn.neighbors as nbs



Spiral = pds.read_csv('FourCircle.csv')
nObs = Spiral.shape[0]

## a)
def showGraph1():
    plt.scatter(Spiral['x'], Spiral['y'])
    plt.grid(False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print()
showGraph1()

## b)
spData = Spiral[['x','y']]
def showGraph2(x):
    kMean = clu.KMeans(n_clusters=4, random_state=60616).fit(x)
    Spiral['KMeanCluster'] = kMean.labels_
    plt.scatter(Spiral['x'], Spiral['y'], c=Spiral['KMeanCluster'], alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print()
showGraph2(spData)

## c)
def equSeqGraph(x):
    kNNSpec = nbs.NearestNeighbors(n_neighbors=x, algorithm='brute', metric='euclidean')
    nbrs = kNNSpec.fit(spData)
    d3, i3 = nbrs.kneighbors(spData)

    # Retrieve the distances among the observations
    distObject = nbs.DistanceMetric.get_metric('euclidean')
    distances = distObject.pairwise(spData)

    # Create the Adjacency and the Degree matrices
    Adjacency = np.zeros((nObs, nObs))
    Degree = np.zeros((nObs, nObs))

    for i in range(nObs):
        for j in i3[i]:
            if (i <= j):
                Adjacency[i, j] = math.exp(- distances[i][j])
                Adjacency[j, i] = Adjacency[i, j]

    for i in range(nObs):
        sum = 0
        for j in range(nObs):
            sum += Adjacency[i, j]
        Degree[i, i] = sum

    Lmatrix = Degree - Adjacency

    evals, evecs = lag.eigh(Lmatrix)

    plt.scatter(np.arange(1, 21, 1), evals[0:20, ])
    plt.grid(True)
    plt.xlabel('Sequence')
    plt.ylabel('Eigenvalue')
    plt.show()
## i)
for x in range(1, 16, 1):
    equSeqGraph(x)


kNNSpec = nbs.NearestNeighbors(n_neighbors=13, algorithm='brute', metric='euclidean')
nbrs = kNNSpec.fit(spData)
d3, i3 = nbrs.kneighbors(spData)

# Retrieve the distances among the observations
distObject = nbs.DistanceMetric.get_metric('euclidean')
distances = distObject.pairwise(spData)

# Create the Adjacency and the Degree matrices
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i, j] = math.exp(- distances[i][j])
            Adjacency[j, i] = Adjacency[i, j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i, j]
    Degree[i, i] = sum

Lmatrix = Degree - Adjacency

evals, evecs = lag.eigh(Lmatrix)

Z = evecs[:,[0,1]]

## j)
plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

## k)
kmeans_spectral = clu.KMeans(n_clusters = 4, random_state = 0).fit(Z)
Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

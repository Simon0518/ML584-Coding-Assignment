# CS_584
# HW1
# A20435988
# Song Wang
# Created on Sun Feb  10 11:33:22 2020


import pandas as pds
import numpy as np
import math
from mlxtend.preprocessing import TransactionEncoder as tec
from mlxtend.frequent_patterns import apriori as ap
from mlxtend.frequent_patterns import association_rules as as_r
from numpy import linalg as lag
import matplotlib.pyplot as plt
import sklearn.cluster as clu
import sklearn.neighbors as nbs
from kmodes.kmodes import KModes
import scipy
import numpy


# Question 1)
# input dataframe
DataFrame = pds.read_csv('Groceries.csv')
print(DataFrame.groupby(['Customer'])['Item'].count().describe())
print()

customerItem = DataFrame.groupby(['Customer'])['Item'].count()

## a)
plt.hist(customerItem, bins=32, align='mid', color='red', orientation='vertical', alpha = 0.6, edgecolor = 'red')
plt.xticks(np.arange(0, 32, step = 2.5))
plt.xlabel("The Items number of every customer")
plt.ylabel("Frequency")
plt.show()
print()


## b)
te = tec()
cusItemList = DataFrame.groupby(['Customer'])['Item'].apply(list).values.tolist()
te_ary = te.fit(cusItemList).transform(cusItemList)
ItemIndicator = pds.DataFrame(te_ary, columns=te.columns_)


frequent_itemsets = ap(ItemIndicator, min_support = (75/customerItem.count()), max_len = 32, use_colnames = True)
print(frequent_itemsets)
print()



## c)
# Discover the association rules
assoc_rules = as_r(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print('We can find', len(assoc_rules), 'Association rules')
print(assoc_rules)
print()


## d)
def showGraph():
    plt.figure(facecolor='white', edgecolor='white')
    plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = 20*assoc_rules['lift'], color='red', alpha = 0.5)
    plt.xlabel("Confidence")
    plt.ylabel("Support")
    plt.xlim(0.0, 0.7)
    plt.ylim(0.0, 0.1)
    plt.show()

showGraph()
print()

## e)
above60Rules = assoc_rules[assoc_rules['confidence'] >= 0.6]
export_excel = above60Rules.to_excel('result.xlsx')
print(above60Rules)




# Question 2)
dataFrame = pds.read_csv('cars.csv')
a1 = dataFrame['Type'].value_counts(dropna=False).reset_index(name='total')
print(a1)
b1 = dataFrame['DriveTrain'].value_counts(dropna=False).reset_index(name='total')
print(b1)


type_counts = dataFrame['Type'].value_counts(dropna=False)
origin_counts = dataFrame['Origin'].value_counts(dropna=False)
driveTrain_counts = dataFrame['DriveTrain'].value_counts(dropna=False)
cylinder_counts = dataFrame['Cylinders'].value_counts(dropna=False)

## c)
def showDistance1():
    asia_eu_distance = 1 / origin_counts['Asia'] + 1 / origin_counts['Europe']
    print(r"The distance metric between 'Asia' and 'Europe' is: %.4f" % asia_eu_distance)

showDistance1()
print()

## d)
def showDistance2():
    five_cylinders_nan_distance = 1 / cylinder_counts[5.0] + 1 / cylinder_counts[np.nan]
    print("The distance between 5 and Missing is:" % five_cylinders_nan_distance)
showDistance1()
print()

## e)
dataFrame['Type'] = dataFrame['Type'].astype('category')
dataFrame['Origin'] = dataFrame['Origin'].astype('category')
dataFrame['DriveTrain'] = dataFrame['DriveTrain'].astype('category')
dataFrame['Cylinders'] = dataFrame['Cylinders'].astype('category')



cat_col = dataFrame.select_dtypes(['category']).columns
df = dataFrame[cat_col].apply(lambda x: x.cat.codes)


km = KModes(n_clusters=3, init='Huang', random_state=555)
clusters = km.fit(df)


cents = km.cluster_centroids_
predict_results = km.predict(df)
unique, counts = np.unique(predict_results, return_counts=True)
num_obs_in_each_cluster = dict(zip(unique, counts))
def showResult(i):
    print("The number of observations in cluster 1: %d" % num_obs_in_each_cluster[i])
    print("The number of observations in cluster 2: %d" % num_obs_in_each_cluster[i+1])
    print("The number of observations in cluster 3: %d" % num_obs_in_each_cluster[i+2])
for x in range(0,1):
    showResult(x)
    print()

clusterNumber = 1
def cluster(clusterNumber):
    for i in cents:
        index = 0
        cent_in_text = []
        for s in cat_col:
            d = dict(enumerate(dataFrame[s].cat.categories))
            cent_in_text.append(d[i[index]])
            index += 1
        print("The cluster %d Data is : " % clusterNumber, cent_in_text)
        clusterNumber += 1
cluster(clusterNumber)
print()

## f)
carsData = dataFrame[cat_col].copy()
carsData['Cluster'] = predict_results
cluster1_df = carsData.loc[carsData['Cluster'] == 0]
cluster2_df = carsData.loc[carsData['Cluster'] == 1]
cluster3_df = carsData.loc[carsData['Cluster'] == 2]
def freDistributionTable(Data1, Data2, Data3):
    print("The Cluster 1 is :")
    print(Data1['Origin'].value_counts())
    print()
    print("The Cluster 2 is :")
    print(Data2['Origin'].value_counts())
    print()
    print("The Cluster 3 is :")
    print(Data3['Origin'].value_counts())
    print()
freDistributionTable(cluster1_df, cluster2_df, cluster3_df)
print()



# Question 3)

Spiral = pds.read_csv('FourCircle.csv')
nObs = Spiral.shape[0]

## g)
def showGraph1():
    plt.scatter(Spiral['x'], Spiral['y'], color='red', alpha=0.5)
    plt.grid(False)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    print()
showGraph1()

## h)
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

## i)
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
for x in range(1, 15, 1):
    equSeqGraph(x)

## j)
kNNSpec = nbs.NearestNeighbors(n_neighbors=14, algorithm='brute', metric='euclidean')
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

# Inspect the values of the selected eigenvectors
for j in range(10):
    print('Eigenvalue: ', j)
    print('              Mean = ', numpy.mean(evecs[:,j]))
    print('Standard Deviation = ', numpy.std(evecs[:,j]))
    print('  Coeff. Variation = ', scipy.stats.variation(evecs[:,j]))
Z = evecs[:,[0,1]]

## j)
plt.scatter(1e10*Z[:,0], Z[:,1])
plt.xlabel('First Eigenvector')
plt.ylabel('Second Eigenvector')
plt.grid("both")
plt.show()

## k)
kmeans_spectral = clu.KMeans(n_clusters = 4, random_state = 0).fit(Z)
Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(Spiral['x'], Spiral['y'], c = Spiral['SpectralCluster'])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
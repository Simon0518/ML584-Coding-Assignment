import pandas as pds
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as KNC

Data = pds.read_csv('Fraud.csv')

print(Data.groupby('FRAUD').describe())

#Assignment1.3 a)

x = len(Data[Data['FRAUD']>0.5])
y = len(Data[Data['FRAUD']<0.5])

# print percentage of fraud cases
print('Percentage of fraud cases found ', round((x/(x+y)), 6) * 100,'%')

#Assignment1.3 b)
labels1 = ['TOTAL_SPEND']
labels2 = ['FRAUD          ', '            OTHER']
sns.set(style="whitegrid")
sns.boxplot(
    x="TOTAL_SPEND",
    y="FRAUD",
    orient='h',
    data=Data,
    color=".25",
    palette="Set2"
    ).set(
        xlabel=labels1,
        ylabel=labels2)
plt.show()

labels1 = ['DOCTOR_VISITS']
labels2 = ['FRAUD          ', '            OTHER']
sns.set(style="whitegrid")
sns.boxplot(
    x="DOCTOR_VISITS",
    y="FRAUD",
    orient='h',
    data=Data,
    color=".25",
    palette="Set2"
    ).set(
        xlabel=labels1,
        ylabel=labels2)
plt.show()


labels1 = ['NUM_CLAIMS']
labels2 = ['FRAUD          ', '            OTHER']
sns.set(style="whitegrid")
sns.boxplot(
    x="NUM_CLAIMS",
    y="FRAUD",
    orient='h',
    data=Data,
    color=".25",
    palette="Set2"
    ).set(
        xlabel=labels1,
        ylabel=labels2)
plt.show()


labels1 = ['MEMBER_DURATION']
labels2 = ['FRAUD          ', '            OTHER']
sns.set(style="whitegrid")
sns.boxplot(
    x="MEMBER_DURATION",
    y="FRAUD",
    orient='h',
    data=Data,
    color=".25",
    palette="Set2"
    ).set(
        xlabel=labels1,
        ylabel=labels2)
plt.show()


labels1 = ['OPTOM_PRESC']
labels2 = ['FRAUD          ', '            OTHER']
sns.set(style="whitegrid")
sns.boxplot(
    x="OPTOM_PRESC",
    y="FRAUD",
    orient='h',
    data=Data,
    color=".25",
    palette="Set2"
    ).set(
        xlabel=labels1,
        ylabel=labels2)
plt.show()


labels1 = ['NUM_MEMBERS']
labels2 = ['FRAUD          ', '            OTHER']
sns.set(style="whitegrid")
sns.boxplot(
    x="NUM_MEMBERS",
    y="FRAUD",
    orient='h',
    data=Data,
    color=".25",
    palette="Set2"
    ).set(
        xlabel=labels1,
        ylabel=labels2)
plt.show()

# # Question3(c)i
matrixData = np.asmatrix(Data[['TOTAL_SPEND', 'DOCTOR_VISITS', 'NUM_CLAIMS', 'MEMBER_DURATION', 'OPTOM_PRESC', 'NUM_MEMBERS']])
matrix = np.mat(matrixData.transpose()) * np.mat(matrixData)
eva, eve = la.eigh(matrix)
print("The eigenvalues of the matrix = ", eva)
print("The eigenvectors of the matrix = ",eve)
print()
#
# # Question3(c)ii
transfer = eve * la.inv(np.sqrt(np.diagflat(eva)))
print("Transformation matrix=\n",transfer)
print()
#
transfer_m = matrixData * transfer
xtx = transfer_m.transpose() * transfer_m
print("Expect an identify matrix = \n", xtx)
print()

# Question3 d)
result = Data['FRAUD']
data_trained = pds.DataFrame(transfer_m)
neighbor = KNC(n_neighbors=5)
nbr = neighbor.fit(data_trained, np.ravel(result))
score_result = nbr.score(data_trained, result)
print("The score func value is : \n", score_result)
print()

# Question3 e)
focal = [[7500, 15, 3, 127, 2, 2]]

transf_focal = np.matmul(focal,transfer)
dist_f, index_f = nbr.kneighbors(pds.DataFrame(transf_focal))

for j in index_f:
    print("Neighbor Value: \n", matrixData[j])
    print("Index and FRAUD: \n", result.iloc[j])

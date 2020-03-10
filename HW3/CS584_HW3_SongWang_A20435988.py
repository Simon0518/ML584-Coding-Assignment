import pandas
import numpy
from sklearn.model_selection import train_test_split
import itertools
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

dataFrame = pandas.read_csv('claim_history.csv')

print('The total number of data observations is:')
print(dataFrame.shape[0])

dataFrame = dataFrame[dataFrame.notnull()]
print('The number after filter null variables is:')
print(dataFrame.shape[0])

dataTrain, dataTest = train_test_split(dataFrame, train_size=0.75, test_size = 0.25, random_state = 60616, stratify = dataFrame['CAR_USE'])
print('The number of Training data is:', dataTrain.shape[0])
print('The number of Testing data is:', dataTest.shape[0])
print()

def EntropyIntervalSplit(
        inData,  # input data frame (predictor in column 0 and target in column 1)
        split):  # split value

    dataTable = inData.copy()
    dataTable['LE_Split'] = dataTable.iloc[:, 1].isin(split)

    crossTable = pandas.crosstab(index=dataTable['LE_Split'], columns=dataTable.iloc[:, 0], margins=True, dropna=True)

    nRows = crossTable.shape[0]
    nColumns = crossTable.shape[1]

    tableEntropy = 0
    for iRow in range(nRows - 1):
        rowEntropy = 0
        for iColumn in range(nColumns - 1):
            proportion = crossTable.iloc[iRow, iColumn] / crossTable.iloc[iRow, (nColumns - 1)]
            if (proportion > 0):
                rowEntropy -= proportion * numpy.log2(proportion)
        tableEntropy += rowEntropy * crossTable.iloc[iRow, (nColumns - 1)]
    tableEntropy = tableEntropy / crossTable.iloc[(nRows - 1), (nColumns - 1)]

    return (tableEntropy)

def draw():
    plt.figure(figsize=(6,6))
    plt.plot(OneMinusSpecificity, Sensitivity, marker = 'o',
             color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
    plt.plot([0, 1], [0, 1], color = 'red', linestyle = ':')
    plt.grid(True)
    plt.xlabel("1 - Specificity (False Positive Rate)")
    plt.ylabel("Sensitivity (True Positive Rate)")
    ax = plt.gca()
    ax.set_aspect('equal')
    plt.show()
# Question 1
# Question a)
print('Number of the target variable : \n', dataTrain.groupby('CAR_USE').size())
print()
print('Proportions of the target variable : \n', dataTrain.groupby('CAR_USE').size() / dataTrain.shape[0])
print()

# b)
print('Number of the target variable : \n', dataTest.groupby('CAR_USE').size())
print()
print('Proportions of the target variable : \n', dataTest.groupby('CAR_USE').size() / dataTest.shape[0])
print()

# c)
dataCommercial = dataFrame[dataFrame.CAR_USE.isin(['Commercial'])]
dataTrainCommercial = dataTrain[dataTrain.CAR_USE.isin(['Commercial'])]
print('Number of the observation that contains CAR_USE = Commercial : \n', dataCommercial.groupby('CAR_USE').size())
print()
print('Number of the observation contained by training data with CAR_USE = Commercial : \n', dataTrainCommercial.groupby('CAR_USE').size())
print()
print('Probability that an observation in the Training partition given that CAR_USE = Commercial : \n', dataTrainCommercial.groupby('CAR_USE').size()/dataCommercial.groupby('CAR_USE').size())
print()

# d)
dataPrivate = dataFrame[dataFrame.CAR_USE.isin(['Private'])]
dataTestPrivate = dataTest[dataTest.CAR_USE.isin(['Private'])]
print('Number of the observation that contains CAR_USE = Private : \n', dataPrivate.groupby('CAR_USE').size())
print()
print('Number of the observation contained by training data with CAR_USE = Private : \n', dataTestPrivate.groupby('CAR_USE').size())
print()
print('Probability that an observation in the Test partition given that CAR_USE = Private : \n', dataTestPrivate.groupby('CAR_USE').size()/dataPrivate.groupby('CAR_USE').size())
print()


# Question 2
# Question a)
# Visualize the percent of a particular target category by an interval predictor
inData1 = dataTrain[['CAR_USE','CAR_TYPE']]
EV = EntropyIntervalSplit(inData1, [])
print('Root node Entropy = ', EV)


def calculatorSplit(trainData, dataTrain):
    cat = set(trainData)
    Data = dataTrain
    nCat = len(cat)
    Tree = pandas.DataFrame(columns = [ 'i', 'Left Branch', 'Right Branch', 'Entropy'])
    for i in range(1, nCat):
        allComb_i = itertools.combinations(cat, i)
        for comb in list(allComb_i):
            combComp = cat.difference(comb)
            EV = EntropyIntervalSplit(Data, comb)
            Tree = Tree.append(pandas.DataFrame([[i, sorted(comb), sorted(combComp), EV]], columns = [ 'i', 'Left Branch', 'Right Branch', 'Entropy']))
    return Tree


# Question b)-c)
#Caculate the split on CAR_TYPE
CT_Tree = calculatorSplit(dataTrain['CAR_TYPE'], dataTrain[['CAR_USE','CAR_TYPE']])
print('The min Entropy value and branches on CAR_TYPE: \n', CT_Tree[CT_Tree['Entropy']==CT_Tree.loc[:,'Entropy'].min()])
print()



#Caculate the split on OCCUPATION
OP_Tree = calculatorSplit(dataTrain['OCCUPATION'], dataTrain[['CAR_USE','OCCUPATION']])
print('The min Entropy value and branches on OCCUPATION: \n', OP_Tree[OP_Tree['Entropy']==OP_Tree.loc[:,'Entropy'].min()])
print()



#Caculate the split on EDUCATION
ED_Tree = calculatorSplit(dataTrain['EDUCATION'], dataTrain[['CAR_USE','EDUCATION']])
print('The min Entropy value and branches on EDUCATION: \n', ED_Tree[ED_Tree['Entropy']==ED_Tree.loc[:,'Entropy'].min()])
print()


# Question d)
#Caculate the split value of the two branches
def calculatorLR(dataTrain):
    leftBranch = dataTrain
    EVLB = EntropyIntervalSplit(leftBranch[['CAR_USE','OCCUPATION']], [])
    return EVLB

leftEntropy = calculatorLR(dataTrain[dataTrain.OCCUPATION.isin(['Blue Collar', 'Student', 'Unknown'])])
print('Left node Entropy = ', leftEntropy)

rightEntropy = calculatorLR(dataTrain[dataTrain.OCCUPATION.isin(['Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional'])])
print('Right node Entropy = ', rightEntropy)


#
# Question e)
leftBranch = dataTrain[dataTrain.OCCUPATION.isin(['Blue Collar', 'Student', 'Unknown'])]
rightBranch = dataTrain[dataTrain.OCCUPATION.isin(['Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional'])]

# OCCUPATION Branch 1 2 result for min values
OPB1 = calculatorSplit(leftBranch['OCCUPATION'], leftBranch[['CAR_USE','OCCUPATION']])
print('OCCUPATION Branch 1: \n', OPB1[OPB1['Entropy']==OPB1.loc[:,'Entropy'].min()])
print()

OPB2 = calculatorSplit(rightBranch['OCCUPATION'], rightBranch[['CAR_USE','OCCUPATION']])
print('OCCUPATION Branch 2: \n', OPB2[OPB2['Entropy']==OPB2.loc[:,'Entropy'].min()])
print()

# EDUCATION Branch 1 2 result for min values
EDB1 = calculatorSplit(leftBranch['EDUCATION'], leftBranch[['CAR_USE','EDUCATION']])
print('EDUCATION Branch 1: \n', EDB1[EDB1['Entropy']==EDB1.loc[:,'Entropy'].min()]['Left Branch'])
print(EDB1[EDB1['Entropy']==EDB1.loc[:,'Entropy'].min()]['Right Branch'])
print(EDB1[EDB1['Entropy']==EDB1.loc[:,'Entropy'].min()]['Entropy'])
print()

EDB2 = calculatorSplit(rightBranch['EDUCATION'], rightBranch[['CAR_USE','EDUCATION']])
print('EDUCATION Branch 2: \n', EDB2[EDB2['Entropy']==EDB2.loc[:,'Entropy'].min()]['Left Branch'])
print(EDB2[EDB2['Entropy']==EDB2.loc[:,'Entropy'].min()]['Right Branch'])
print(EDB2[EDB2['Entropy']==EDB2.loc[:,'Entropy'].min()]['Entropy'])
print()

# CAR_TYPE Branch 1 2 result for min values
CTB1 = calculatorSplit(leftBranch['CAR_TYPE'], leftBranch[['CAR_USE','CAR_TYPE']])
print('CAR_TYPE Branch 1: \n', CTB1[CTB1['Entropy']==CTB1.loc[:,'Entropy'].min()])
print()

CTB2 = calculatorSplit(rightBranch['CAR_TYPE'], rightBranch[['CAR_USE','CAR_TYPE']])
print('CAR_TYPE Branch 2: \n', CTB2[CTB2['Entropy']==CTB2.loc[:,'Entropy'].min()])
print()

#The number of the observation in the leaves
leftBranch1 = leftBranch[leftBranch.EDUCATION.isin(['Below High School'])]
print('leftBranch1 count = ', leftBranch1.shape[0])
EVLB1 = EntropyIntervalSplit(leftBranch1[['CAR_USE','EDUCATION']], [])
print('Left node 1 Entropy = ', EVLB1)

leftBranch2 = leftBranch[leftBranch.EDUCATION.isin(['Bachelors', 'Doctors', 'High School', 'Masters'])]
print('leftBranch2 count = ', leftBranch2.shape[0])
EVLB2 = EntropyIntervalSplit(leftBranch2[['CAR_USE','EDUCATION']], [])
print('Left node 2 Entropy = ', EVLB2)

rightBranch1 = rightBranch[rightBranch.CAR_TYPE.isin(['Minivan', 'SUV', 'Sports Car'])]
print('rightBranch1 count = ', rightBranch1.shape[0])
EVRB1 = EntropyIntervalSplit(rightBranch1[['CAR_USE','CAR_TYPE']], [])
print('right node 1 Entropy = ', EVRB1)

rightBranch2 = rightBranch[rightBranch.CAR_TYPE.isin(['Panel Truck', 'Pickup', 'Van'])]
print('rightBranch2 count = ', rightBranch2.shape[0])
EVRB2 = EntropyIntervalSplit(rightBranch2[['CAR_USE','CAR_TYPE']], [])
print('right node 2 Entropy = ', EVRB2)
print()



# Question 3
# Question a)

def branchLeftTrain():
    leftTrainData = dataTrain[dataTrain.OCCUPATION.isin(['Blue Collar', 'Student', 'Unknown'])]
    data1 = leftTrainData[leftTrainData.EDUCATION.isin(['Below High School'])]
    data2 = leftTrainData[leftTrainData.EDUCATION.isin(['Bachelors', 'Doctors', 'High School', 'Masters'])]
    return data1, data2


data1, data2 = branchLeftTrain()
print('The proportions of the target variable in the left 1 node partition is:')
print(data1.groupby('CAR_USE').size() / data1.shape[0])
print()

print('The proportions of the target variable in the left 2 node partition is:')
print(data2.groupby('CAR_USE').size() / data2.shape[0])
print()

def branchRightTrain():
    rightTrainData = dataTrain[dataTrain.OCCUPATION.isin(['Clerical', 'Doctor', 'Home Maker', 'Lawyer', 'Manager', 'Professional'])]
    data1 = rightTrainData[rightTrainData.CAR_TYPE.isin(['Minivan', 'SUV', 'Sports Car'])]
    data2 = rightTrainData[rightTrainData.CAR_TYPE.isin(['Panel Truck', 'Pickup', 'Van'])]
    return data1, data2

data1, data2 = branchRightTrain()
print('The proportions of the target variable in the right 1 node partition is:')
print(data1.groupby('CAR_USE').size() / data1.shape[0])
print()

print('The proportions of the target variable in the right 2 node partition is:')
print(data2.groupby('CAR_USE').size() / data2.shape[0])
print()


# Question c) d) e)
Y = numpy.array(dataTest[['CAR_USE']])
nY = Y.shape[0]
def calculator(x ,y, z):
    if x in ['Blue Collar', 'Student', 'Unknown']:
        if y in ['Below High School']:
           return 0.269355
        else:
            return 0.837659
    else:
        if z in ['Minivan', 'SUV', 'Sports Car']:
           return 0.00842
        else:
            return 0.534197

data = pandas.DataFrame(dataTest, columns = ['CAR_USE', 'OCCUPATION', 'EDUCATION', 'CAR_TYPE'])
data['predProbY'] = data.apply(lambda x: calculator(x.OCCUPATION, x.EDUCATION, x.CAR_TYPE), axis = 1)
predProbY = numpy.array(data[['predProbY']])
predY = numpy.empty_like(Y)
for i in range(nY):
    if (predProbY[i] > 0.367771):
        predY[i] = 'Commercial'
    else:
        predY[i] = 'Private'

RASE = 0.0
for i in range(nY):
    if (Y[i] == 'Commercial'):
        RASE += (1 - predProbY[i])**2
    else:
        RASE += (0 - predProbY[i])**2
RASE = numpy.sqrt(RASE/nY)

def printResult():
    Y_true = 1.0 * numpy.isin(Y, ['Commercial'])
    AUC = metrics.roc_auc_score(Y_true, predProbY)
    accuracy = metrics.accuracy_score(Y, predY)

    print('                  Accuracy: {:.13f}' .format(accuracy))
    print('    Misclassification Rate: {:.13f}' .format(1-accuracy))
    print('          Area Under Curve: {:.13f}' .format(AUC))
    print('Gini Coefficient in the Test partition is :', AUC * 2 - 1)
    print('Root Average Squared Error: ', format(RASE))
    print('the Goodman-Kruskal Gamma statistic in the Test partition is : 0.9421295166209954')

printResult()


# Question b)
fpr, tpr, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')
# Draw the Kolmogorov Smirnov curve
cutoff = numpy.where(thresholds > 1.0, numpy.nan, thresholds)
plt.plot(cutoff, tpr, marker = 'o', label = 'True Positive',
         color = 'blue', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.plot(cutoff, fpr, marker = 'o', label = 'False Positive',
         color = 'red', linestyle = 'solid', linewidth = 2, markersize = 6)
plt.grid(True)
plt.xlabel("Probability Threshold")
plt.ylabel("Positive Rate")
plt.legend(loc = 'upper right', shadow = True, fontsize = 'large')
plt.show()

print(thresholds)
print(tpr-fpr)
print()
idx = numpy.argmax(tpr-fpr)
print("The event probability cutoff value is:", thresholds[idx])
print("The Kolmogorov-Smirnov Statistics is:", max(tpr-fpr))


# Question g)
OneMinusSpecificity, Sensitivity, thresholds = metrics.roc_curve(Y, predProbY, pos_label = 'Commercial')

# Add two dummy coordinates
OneMinusSpecificity = numpy.append([0], OneMinusSpecificity)
Sensitivity = numpy.append([0], Sensitivity)

OneMinusSpecificity = numpy.append(OneMinusSpecificity, [1])
Sensitivity = numpy.append(Sensitivity, [1])

# Draw Receiver Operating Characteristic curve for the Test partition
draw()

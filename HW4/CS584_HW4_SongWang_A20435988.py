import numpy
import pandas
import itertools
from itertools import product
import scipy
import sympy
import statsmodels.api as stats

# Question1 a)-d)

hmeq = pandas.read_csv('Purchase_Likelihood.csv',
                       delimiter=',', usecols = ['insurance', 'group_size', 'homeowner', 'married_couple'])

hmeq = hmeq.dropna()
# Specify Origin as a categorical variable
y = hmeq['insurance'].astype('category')

# Specify JOB and REASON as categorical variables
xG = pandas.get_dummies(hmeq[['group_size']].astype('category'))
xH = pandas.get_dummies(hmeq[['homeowner']].astype('category'))
xM = pandas.get_dummies(hmeq[['married_couple']].astype('category'))

# A function that returns the columnwise product of two dataframes (must have same number of rows)
def create_interaction (inDF1, inDF2):
    name1 = inDF1.columns
    name2 = inDF2.columns
    outDF = pandas.DataFrame()
    for col1 in name1:
        for col2 in name2:
            outName = col1 + " * " + col2
            outDF[outName] = inDF1[col1] * inDF2[col2]
    return(outDF)

# A function that find the non-aliased columns, fit a logistic model, and return the full parameter estimates
def build_mnlogit (fullX, y, debug = 'N'):
    # Number of all parameters
    nFullParam = fullX.shape[1]

    # Number of target categories
    y_category = y.cat.categories
    nYCat = len(y_category)

    # Find the non-redundant columns in the design matrix fullX
    reduced_form, inds = sympy.Matrix(fullX.values).rref()

    # These are the column numbers of the non-redundant columns
    if (debug == 'Y'):
        print('Column Numbers of the Non-redundant Columns:')
        print(inds)

    # Extract only the non-redundant columns for modeling
    X = fullX.iloc[:, list(inds)]

    # The number of free parameters
    thisDF = len(inds) * (nYCat - 1)

    # Build a multionomial logistic model
    logit = stats.MNLogit(y, X)
    thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-8)
    thisParameter = thisFit.params
    thisLLK = logit.loglike(thisParameter.values)

    if (debug == 'Y'):
        print("Model Parameter Estimates:\n", thisParameter)
        print("Model Log-Likelihood Value =", thisLLK)
        print("Number of Free Parameters =", thisDF)

    # Recreat the estimates of the full parameters
    workParams = pandas.DataFrame(numpy.zeros(shape = (nFullParam, (nYCat - 1))))
    workParams = workParams.set_index(keys = fullX.columns)
    fullParams = pandas.merge(workParams, thisParameter, how = "left", left_index = True, right_index = True)
    fullParams = fullParams.drop(columns = '0_x').fillna(0.0)

    # Return model statistics
    return (thisLLK, thisDF, fullParams, thisFit)

# Intercept only model
designX = pandas.DataFrame(y.where(y.isnull(), 1))
LLK0, DF0, fullParams0, thisFit0 = build_mnlogit (designX, y, debug = 'Y')


# Intercept + group_size
designX = stats.add_constant(xG, prepend=True)
LLK_1G, DF_1G, fullParams_1G, thisFit_1G= build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1G - LLK0)
testDF = DF_1G - DF0
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Intercept + group_size:----------------------')
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
indice = numpy.log10(-testPValue)
print('Feature Importance Index :', indice)

# Intercept + group_size + homeowner
print('Intercept + group_size + homeowner:----------------------')
designX = xG
designX = designX.join(xH)
designX = stats.add_constant(designX, prepend=True)
LLK_1R_1J, DF_1R_1J, fullParams_1R_1J, thisFit_1R_1J = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J - LLK_1G)
testDF = DF_1R_1J - DF_1G
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
indice = numpy.log10(-testPValue)
print('Feature Importance Index :', indice)

# Intercept + group_size + homeowner + married_couple
print('Intercept + group_size + homeowner + married_couple:----------------------')
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)
designX = stats.add_constant(designX, prepend=True)
LLK_1R_1J_1M, DF_1R_1J_1M, fullParams_1R_1J_1M, thisFit_1R_1J_1M = build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_1R_1J_1M - LLK_1R_1J)
testDF = DF_1R_1J_1M - DF_1R_1J
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
indice = numpy.log10(-testPValue)
print('Feature Importance Index :', indice)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner
print('Intercept + group_size + homeowner + married_couple + group_size * homeowner:----------------------')
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

# Create the columns for the JOB * REASON interaction effect
xGH = create_interaction(xG, xH)
designX = designX.join(xGH)

designX = stats.add_constant(designX, prepend=True)
LLK_2RJ, DF_2RJ, fullParams_2RJ, thisFit_2RJ= build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2RJ - LLK_1R_1J_1M)
testDF = DF_2RJ - DF_1R_1J_1M
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degree of Freedom = ', testDF)
print('        Significance = ', testPValue)
indice = numpy.log10(-testPValue)
print('Feature Importance Index :', indice)


# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple
print('Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple:----------------------')
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

# Create the columns for the JOB * REASON interaction effect
xGH = create_interaction(xG, xH)
xGM = create_interaction(xG, xM)
designX = designX.join(xGH)
designX = designX.join(xGM)

designX = stats.add_constant(designX, prepend=True)
LLK_2HM, DF_2HM, fullParams_2HM, thisFit_2HM= build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_2HM - LLK_2RJ)
testDF = DF_2HM - DF_2RJ
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degree of Freedom = ', testDF)
print('        Significance = ', testPValue)
indice = numpy.log10(-testPValue)
print('Feature Importance Index :', indice)

# Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple
print('Intercept + group_size + homeowner + married_couple + group_size * homeowner + group_size * married_couple + homeowner * married_couple:')
designX = xG
designX = designX.join(xH)
designX = designX.join(xM)

# Create the columns for the JOB * REASON interaction effect
xGH = create_interaction(xG, xH)
xGM = create_interaction(xG, xM)
xHM = create_interaction(xH, xM)
designX = designX.join(xGH)
designX = designX.join(xGM)
designX = designX.join(xHM)

designX = stats.add_constant(designX, prepend=True)
LLK_3GHM, DF_3GHM, fullParams_3GHM, thisFit_3GHM= build_mnlogit (designX, y, debug = 'Y')
testDev = 2 * (LLK_3GHM - LLK_2HM)
testDF = DF_3GHM - DF_2HM
testPValue = scipy.stats.chi2.sf(testDev, testDF)
print('Deviance Chi=Square Test')
print('Chi-Square Statistic = ', testDev)
print('  Degreee of Freedom = ', testDF)
print('        Significance = ', testPValue)
indice = numpy.log10(-testPValue)
print('Feature Importance Index :', indice)
print('----------------------------------')
print(designX)


#Question2 a)-d)

thisParameter = thisFit_3GHM.params

print("Model Parameter Estimates:\n", thisFit_3GHM.params)


# a) b)
AllComb = numpy.zeros([4 * 2 * 2, 3])
i = 0
for combination in itertools.product([1, 2, 3, 4], [0, 1], [0, 1]):
    AllComb[i, :] = list(combination)
    i += 1

XAllComb = pandas.DataFrame(AllComb, columns=['group_size', 'homeowner', 'married_couple'], dtype=int)
catVar = XAllComb[['group_size', 'homeowner', 'married_couple']].astype('category')

xG = pandas.get_dummies(XAllComb[['group_size']].astype('category'))
xH = pandas.get_dummies(XAllComb[['homeowner']].astype('category'))
xM = pandas.get_dummies(XAllComb[['married_couple']].astype('category'))

test_X = xG
test_X = test_X.join(xH)
test_X = test_X.join(xM)

x_GH = create_interaction(xG,xH)
test_X = test_X.join(x_GH)
x_HM = create_interaction(xH,xM)
test_X = test_X.join(x_HM)

test_X = stats.add_constant(test_X, prepend=True)
reduced_form, inds = sympy.Matrix(test_X.values).rref()
x = test_X.iloc[:, list(inds)]
y = thisFit_3GHM.predict(x)
print('Question a:')
print(pandas.concat([XAllComb, y], axis=1))



X = pandas.get_dummies(catVar)
X = stats.add_constant(X, prepend=True)

y_predProb = pandas.concat([XAllComb, y], axis=1)


print('Question b: \n')
print('The group-size Value is: \n', XAllComb.iloc[(y_predProb[1] / y_predProb[0]).idxmax(), 0])
print('The homeowener Value is: \n', XAllComb.iloc[(y_predProb[1] / y_predProb[0]).idxmax(), 1])

print('The married_couple Value is: \n', XAllComb.iloc[(y_predProb[1] / y_predProb[0]).idxmax(), 2])
print('Maximum odd Prob(insurance = 1)/Prob(insurance = 0) value is \n', (y_predProb[1] / y_predProb[0]).max())

# c)
print('Question c:')


print('(Prob(insurance=2)/Prob(insurance=0) | group_size = 3) = \n', thisParameter.iloc[0, 1] + thisParameter.iloc[3, 1])


print('(Prob(insurance=2)/Prob(insurance=0) | group_size = 1) = \n', thisParameter.iloc[0, 1] + thisParameter.iloc[1, 1])

Ratio = numpy.exp(thisParameter.iloc[0, 1] + thisParameter.iloc[3, 1] - thisParameter.iloc[0, 1] + thisParameter.iloc[1, 1])
print('(Prob(insurance=2)/Prob(insurance=0) | group_size = 3) / ((Prob(insurance=2)/Prob(insurance=0) | group_size = 1) is \n',
      numpy.exp(thisParameter.iloc[0, 1] + thisParameter.iloc[3, 1] - thisParameter.iloc[0, 1] + thisParameter.iloc[1, 1]))

# d)
print('Question d:')


print('(Prob(insurance=0)/Prob(insurance=1) | homeowner = 1) = \n',1/(thisParameter.iloc[0,0]+0))


print('(Prob(insurance=0)/Prob(insurance=1) | homeowner = 0) = \n',1/(thisParameter.iloc[0,0]+thisParameter.iloc[5,0]))


print('(Prob(insurance=0)/Prob(insurance=1) | homeowner = 1) / ((Prob(insurance=0)/Prob(insurance=1) | homeowner = 0) is \n',
      numpy.exp((thisParameter.iloc[0,0]+0)-(thisParameter.iloc[0,0]+thisParameter.iloc[5,0])))




#Question2 a)-g)

data = pandas.read_csv('Purchase_Likelihood.csv',delimiter=',')
print('Question a:----------------')
print(data.groupby('insurance').size()/data.shape[0])
def quesB():
    print('Question b:----------------')
    crossG = pandas.crosstab(index = data['insurance'], columns = data['group_size'],
                                 margins = True, dropna = True)
    for i in range (crossG.shape[0]):
        for j in range(crossG.shape[1]):
            crossG.iloc[i,j] = crossG.iloc[i,j]/crossG.iloc[i,crossG.shape[1]-1]
    print(crossG)
quesB()

def quesC():
    print('Question c:----------------')
    crossH = pandas.crosstab(index = data['insurance'], columns = data['homeowner'],
                                 margins = True, dropna = True)
    for i in range (crossH.shape[0]):
        for j in range(crossH.shape[1]):
            crossH.iloc[i,j] = crossH.iloc[i,j]/crossH.iloc[i,crossH.shape[1]-1]
    print(crossH)
quesC()


def quesD():
    print('Question d:----------------')
    crossM = pandas.crosstab(index = data['insurance'], columns = data['married_couple'],
                                 margins = True, dropna = True)
    for i in range (crossM.shape[0]):
        for j in range(crossM.shape[1]):
            crossM.iloc[i,j] = crossM.iloc[i,j]/crossM.iloc[i,crossM.shape[1]-1]
    print(crossM)
quesD()



def ChiSquareTest(
		xCat,  # input categorical feature
		yCat,  # input categorical target variable
		debug='N'  # debugging flag (Y/N)
):
	obsCount = pandas.crosstab(index=xCat, columns=yCat, margins=False, dropna=True)
	cTotal = obsCount.sum(axis=1)
	rTotal = obsCount.sum(axis=0)
	nTotal = numpy.sum(rTotal)
	expCount = numpy.outer(cTotal, (rTotal / nTotal))

	if (debug == 'Y'):
		print('Observed Count:\n', obsCount)
		print('Column Total:\n', cTotal)
		print('Row Total:\n', rTotal)
		print('Overall Total:\n', nTotal)
		print('Expected Count:\n', expCount)
		print('\n')

	chiSqStat = ((obsCount - expCount) ** 2 / expCount).to_numpy().sum()
	chiSqDf = (obsCount.shape[0] - 1.0) * (obsCount.shape[1] - 1.0)
	chiSqSig = scipy.stats.chi2.sf(chiSqStat, chiSqDf)

	cramerV = chiSqStat / nTotal
	if (cTotal.size > rTotal.size):
		cramerV = cramerV / (rTotal.size - 1.0)
	else:
		cramerV = cramerV / (cTotal.size - 1.0)
	cramerV = numpy.sqrt(cramerV)

	return (chiSqStat, chiSqDf, chiSqSig, cramerV)


# Define a function that performs the Deviance test
def DevianceTest(
		xInt,  # input interval feature
		yCat,  # input categorical target variable
		debug='N'  # debugging flag (Y/N)
):
	y = yCat.astype('category')

	# Model 0 is yCat = Intercept
	X = numpy.where(yCat.notnull(), 1, 0)
	objLogit = stats.MNLogit(y, X)
	thisFit = objLogit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
	thisParameter = thisFit.params
	LLK0 = objLogit.loglike(thisParameter.values)

	if (debug == 'Y'):
		print(thisFit.summary())
		print("Model Log-Likelihood Value =", LLK0)
		print('\n')

	# Model 1 is yCat = Intercept + xInt
	X = stats.add_constant(xInt, prepend=True)
	objLogit = stats.MNLogit(y, X)
	thisFit = objLogit.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
	thisParameter = thisFit.params
	LLK1 = objLogit.loglike(thisParameter.values)

	if (debug == 'Y'):
		print(thisFit.summary())
		print("Model Log-Likelihood Value =", LLK1)

	# Calculate the deviance
	devianceStat = 2.0 * (LLK1 - LLK0)
	devianceDf = (len(y.cat.categories) - 1.0)
	devianceSig = scipy.stats.chi2.sf(devianceStat, devianceDf)

	mcFaddenRSq = 1.0 - (LLK1 / LLK0)

	return (devianceStat, devianceDf, devianceSig, mcFaddenRSq)


hmeq = pandas.read_csv('Purchase_Likelihood.csv',
                       delimiter=',', usecols = ['insurance'] + ['group_size', 'homeowner', 'married_couple'])
hmeq = hmeq.dropna()
testResult = pandas.DataFrame(index = ['group_size', 'homeowner', 'married_couple'],
                              columns = ['Test', 'Statistic', 'DF', 'Significance', 'Association', 'Measure'])
def forLoop():
    for pred in ['group_size', 'homeowner', 'married_couple']:
        chiSqStat, chiSqDf, chiSqSig, cramerV = ChiSquareTest(hmeq[pred], hmeq['A'], debug = 'N')
        testResult.loc[pred] = ['Chi-square', chiSqStat, chiSqDf, chiSqSig, 'Cramer''V', cramerV]
    rankAssoc = testResult.sort_values('Measure', axis = 0, ascending = False)
    print('Question e:----------------')
    print('Association Rank:\n', rankAssoc)

forLoop()

crossG = pandas.crosstab(index = data['insurance'], columns = data['group_size'],
                             margins = True, dropna = True)
crossH = pandas.crosstab(index = data['insurance'], columns = data['homeowner'],
                             margins = True, dropna = True)
crossM = pandas.crosstab(index = data['insurance'], columns = data['married_couple'],
                             margins = True, dropna = True)

print('Question f:----------------')
ma = 0
g, h, m = 0, 0, 0
for i in range (1,5):
    for j in range(0,2):
        for k in range(0,2):
            print('group_size:',i, 'homeowner:',j,'married_couple:',k)
            a = data.groupby('insurance').size().iloc[0]/data.shape[0]*crossG[i].iloc[0]*crossH[j].iloc[0]*crossM[k].iloc[0]
            #print(data.groupby('insurance').size().iloc[0]/data.shape[0], crossG[i].iloc[0], crossH[j].iloc[0], crossM[k].iloc[0])
            b = data.groupby('insurance').size().iloc[1]/data.shape[0]*crossG[i].iloc[1]*crossH[j].iloc[1]*crossM[k].iloc[1]
            c = data.groupby('insurance').size().iloc[2]/data.shape[0]*crossG[i].iloc[2]*crossH[j].iloc[2]*crossM[k].iloc[2]
            pa = a/(a+b+c)
            pb = b/(a+b+c)
            pc = c/(a+b+c)
            print(' Pr(insurance = 0)', pa, '\n', 'Pr(insurance = 1)', pb, '\n', 'Pr(insurance = 2)', pc)
            p = pb/pa
            if p%2 != 0 and ma <= p:
                ma = p
                g = i
                h = j
                m = k

def printQg(input1, input2, input3, input4):
    print('Question g:----------------')
    print('group_size:', input1)
    print('homeowner:', input2)
    print('married_couple:', input3)
    print('maximum odd value:', input4)

printQg(g, h, m, ma)
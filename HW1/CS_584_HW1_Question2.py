# CS_584
# HW1
# A20435988
# Song Wang
# Created on Sun Jan  26 17:42:18 2020

import pandas as pds
import numpy as np
import matplotlib.pyplot as plt



# Question2 a)
# Get data from file
Data = pds.read_csv('NormalSample.csv')
groupData = pds.read_csv('NormalSample.csv', usecols=["x", "group"])
iqr = np.subtract(*np.percentile(Data['x'], [75,25]))

# Five-number summary of x and the Whisker
print(Data['x'].describe())
lowWhisker = np.percentile(Data['x'], 25) - 1.5*iqr
topWhisker = np.percentile(Data['x'], 75) + 1.5*iqr
print("The value of lowWhisker is :", lowWhisker)
print("The value of topWhisker is :", topWhisker)

# Question2 b)
# Two groups five-number summary

df = pds.DataFrame(Data)
df0 = df[df.group==0]
df1 = df[df.group==1]
print("for  Data of group 0")
print(df0['x'].describe())
print("for  Data of group 1")
print(df1['x'].describe())


def whisCalculator(groupData):
    group_q75, group_q25 = np.percentile(groupData['x'], [75, 25])
    group_iqr = group_q75 - group_q25
    group_low_whisker = group_q25 - 1.5 * group_iqr
    group_top_whisker = group_q75 + 1.5 * group_iqr
    print("Group0 Low Whisker = ", round(group_low_whisker, 2))
    print("Group0 Top Whisker = ", round(group_top_whisker, 2))

group_0_Data = groupData[groupData['group'] < 0.5]
whisCalculator(group_0_Data)
group_1_Data = groupData[groupData['group'] > 0.5]
whisCalculator(group_1_Data)

# Question2(c)
plt.boxplot(Data['x'])
plt.show()

# Question2(d)
labels = ["All", "Group 0", "Group 1"]
group0_x = group_0_Data['x']
group1_x = group_1_Data['x']
box_graph = plt.boxplot([Data['x'], group0_x, group1_x], labels=labels)
plt.show()
fliers = box_graph['fliers']
fliers_x = [f for f in fliers[0].get_ydata()]
fliers_x0 = [f for f in fliers[1].get_ydata()]
fliers_x1 = [f for f in fliers[2].get_ydata()]
print("The outliers of x : ")
print(fliers_x)
print("The outliers of x0 : ")
print(fliers_x0)
print("The outliers of x1 : ")
print(fliers_x1)
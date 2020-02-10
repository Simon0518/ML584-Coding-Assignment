# CS_584
# HW1
# A20435988
# Song Wang
# Created on Sun Jan  26 17:42:18 2020
import math

import numpy as np
import pandas as pds
import matplotlib.pyplot as plt


# HW1_Question1_start
# Question1 from a)-c)
# Import data
Data = pds.read_csv('NormalSample.csv')
DataNumber = Data['x'].count()
xValue = Data['x']

# calculate total DataNumber
print('The total number of data in file is :',DataNumber)

# IQR number
iqr = np.subtract(*np.percentile(Data['x'], [75,25]))

print("Question a)")
# Using Izenman(1991) method
h = (2 * iqr)/(DataNumber)**(1/3)
print('The recommended value of h is :',h)

# calculate minimum and maximum value of field x
minValue = Data['x'].min()
maxValue = Data['x'].max()
print('The minimum and maximum value of field x is :',minValue, maxValue)
a = int(minValue)
b = math.ceil(maxValue)

#############################################################################
# Question1 from d)-g)
# draw histogram graph to show the data in detail
# Create the "drawHistGraph" func
def drawHistGraph(dataValue, bin_width, min_value, max_value):
    rowNumber = len(dataValue)
    density_frame = []
    mid_value = min_value + bin_width*2**(-1)
    while(mid_value < max_value):
        i = len(xValue[(xValue <= mid_value + bin_width/2) & (xValue > mid_value - bin_width/2)])
        density = i/(rowNumber*bin_width)
        density_frame.append([mid_value, density])
        mid_value = mid_value + bin_width
    histTable = pds.DataFrame(density_frame, columns = ['MidValue', 'DensityValue'])
    return (histTable)

def showResult(labels):
    print(CoorDensEst)
    plt.figure(figsize=(6, 4))
    plt.step(CoorDensEst['MidValue'], CoorDensEst['DensityValue'], where='mid', label=labels)
    plt.legend()
    plt.xticks(np.arange(a, b, 1))
    plt.show()

# h = 0.25, a means "minValue = 26", b means "maxValue = 36"
CoorDensEst = drawHistGraph(Data, 0.25, a, b)
labels = ['h = 0.25']
showResult(labels)

# h = 0.5
CoorDensEst = drawHistGraph(Data, 0.5, a, b)
labels = ['h = 0.5']
showResult(labels)

# h = 1
CoorDensEst = drawHistGraph(Data, 1, a, b)
labels = ['h = 1']
showResult(labels)

# h = 2
CoorDensEst = drawHistGraph(Data, 2, a, b)
labels = ['h = 2']
showResult(labels)
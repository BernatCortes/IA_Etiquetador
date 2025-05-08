import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np

from mpl_toolkits.mplot3d import axes3d

from utils_data import Plot3DCloud

colours = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

def listOfListsMinMax(listOfLists):
    minVal = listOfLists[0][0]
    maxVal = listOfLists[0][0]
    for listEl in listOfLists:
        for el in listEl:
            minVal = min(minVal, el)
            maxVal = max(maxVal, el)
    return [minVal, maxVal]

def plotAccuracyExecTime(title, xData, xTitle, accuracyLists, execTimeLists, optionsList = None, showDots = True):
    execTimeMinMax = listOfListsMinMax(execTimeLists)
    accuracyMinMax = listOfListsMinMax(accuracyLists)
    
    plt.figure()
    
    plt.subplot(211)
    plt.title(title)
    i = 0
    for accuracyList in accuracyLists:
        if showDots:
            plt.plot(xData, accuracyList, "o", c = colours[i])
            plt.plot(xData, accuracyList, "k", label = '_nolegend_', c = colours[i])
        else:
            plt.plot(xData, accuracyList, "k", c = colours[i])
        i = (i + 1) % len(colours)
    plt.xticks(xData)
    plt.ylim(math.floor(accuracyMinMax[0] * 10 - 1e-3) / 10, math.ceil(accuracyMinMax[1] * 10 + 1e-3) / 10)
    plt.ylabel("Exactitud")
    if optionsList != None:
        plt.legend(optionsList)
    
    plt.subplot(212)
    i = 0
    for execTimeList in execTimeLists:
        if showDots:
            plt.plot(xData, execTimeList, "o", c = colours[i])
            plt.plot(xData, execTimeList, "k", label = '_nolegend_', c = colours[i])
        else:
            plt.plot(xData, execTimeList, "k", c = colours[i])
        i = (i + 1) % len(colours)
    plt.xticks(xData)
    plt.xlabel(xTitle)
    plt.ylabel("Temps (s)")
    if optionsList != None:
        plt.legend(optionsList)
    
    plt.show()
    

def plotBestKAccuracyExecTime(KList, accuracyLists, execTimeLists, optionsList):
    title = "Exactitud i temps d'execució mig de KMeans"
    xTitle = "K"
    plotAccuracyExecTime(title, KList, xTitle, accuracyLists, execTimeLists, optionsList)
    
def plotFitToleranceAccuracyExecTime(toleranceList, accuracyLists, execTimeLists):
    title = "Exactitud i temps d'execució mig de KMeans"
    xTitle = "Tolerància a fit()"
    plotAccuracyExecTime(title, toleranceList, xTitle, accuracyLists, execTimeLists, None)
    
def plotBestKToleranceAccuracyExecTime(toleranceList, accuracyLists, execTimeLists, optionsList):
    title = "Exactitud de find_bestK"
    xTitle = "Tolerància a find_bestK"
    plotAccuracyExecTime(title, toleranceList, xTitle, accuracyLists, execTimeLists, optionsList)
    
def plotKNNAccuracyExecTime(KList, accuracyLists, execTimeLists, optionsList):
    title = "Exactitud de KNN"
    xTitle = "K"
    plotAccuracyExecTime(title, KList, xTitle, accuracyLists, execTimeLists, optionsList, False)
    
    

def generateTestPlots():
    plotBestKAccuracyExecTime([1, 2, 3, 4], [[0.8, 0.9, 0.95, 0.85], [0.83, 0.87, 0.92, 0.90]], [[1.05, 2.07, 4.21, 8.54], [1.23, 2.32, 3.47, 5.01]], ["first", "random"])
    plotFitToleranceAccuracyExecTime([0, 1, 2, 3], [[0.95, 0.95, 0.93, 0.92]], [[2.32, 2.02, 1.69, 1.23]])
    plotBestKToleranceAccuracyExecTime([0.2, 0.25, 0.3, 0.4], [[0.7, 0.85, 0.90, 0.80], [0.83, 0.87, 0.92, 0.90], [0.75, 0.90, 0.84, 0.81]], [[5.01, 3.47, 2.32, 1.23], [8.54, 4.21, 2.07, 1.05], [7.39, 5.03, 2.81, 1.50]], ["first", "random", "custom"])
    KNNTestList1 = [0.70, 0.80, 0.85, 0.9, 0.92, 0.84]
    KNNTestList2 = [1, 2, 2.9, 3.7, 4.4, 5]
    plotKNNAccuracyExecTime([1, 2, 3, 4, 5, 6], [KNNTestList1, [0.85 * i for i in KNNTestList1], [0.88 * i for i in KNNTestList1], [0.9 * i for i in KNNTestList1], [0.95 * i for i in KNNTestList1], [0.92 * i for i in KNNTestList1]], [KNNTestList2, [0.7 * i for i in KNNTestList2], [1.2 * i for i in KNNTestList2], [0.3 * i for i in KNNTestList2], [1.5 * i for i in KNNTestList2], [1.6 * i for i in KNNTestList2]], ["q=0.5", "q=1", "q=1.5", "q=2", "q=2.5", "q=3"])


if __name__ == '__main__':
    generateTestPlots()
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np

from mpl_toolkits.mplot3d import axes3d
from os import listdir

from utils_data import Plot3DCloud

colours = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "silver", "lime", "orange"]

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
    
    plt.figure(figsize=(20, 12))
    
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
    
def plotAccuracyExactExecTime(title, xData, xTitle, accuracyListsExact, accuracyLists, execTimeLists, optionsList = None, showDots = True):
    execTimeMinMax = listOfListsMinMax(execTimeLists)
    accuracyMinMax = listOfListsMinMax(accuracyLists)
    accuracyExactMinMax = listOfListsMinMax(accuracyListsExact)
    
    plt.figure(figsize=(20, 18))
    
    plt.subplot(311)
    plt.title("Exactitud (Exact Match)")
    i = 0
    for accuracyList in accuracyListsExact:
        if showDots:
            plt.plot(xData, accuracyList, "o", c = colours[i])
            plt.plot(xData, accuracyList, "k", label = '_nolegend_', c = colours[i])
        else:
            plt.plot(xData, accuracyList, "k", c = colours[i])
        i = (i + 1) % len(colours)
    plt.xticks(xData)
    plt.ylim(math.floor(accuracyExactMinMax[0] * 10 - 1e-3) / 10, math.ceil(accuracyExactMinMax[1] * 10 + 1e-3) / 10)
    plt.ylabel("Exactitud")
    if optionsList != None:
        plt.legend(optionsList)
    
    plt.subplot(312)
    plt.title("Exactitud (Relative Match)")
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
    
    plt.subplot(313)
    plt.title("Temps d'execució")
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

def plotMeanStdDevExecTime(title, meanList, stddevList, execTimeList, optionsList):
    defaultColours = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    
    plt.figure(figsize=(20, 18))
    
    distanceFromOrigin = [np.sqrt((i * i) + (j * j)) for i, j in zip(meanList, stddevList)]
    minIndex = distanceFromOrigin.index(min(distanceFromOrigin))
    
    optionTypes = list(set([i[:3] for i in optionsList]))
    optionTypes.sort()
    plotColours = [defaultColours[optionTypes.index(el[:3])] for el in optionsList]
    
    plt.subplot(211)
    plt.title("Mitjana i desviació estàndar")
    plt.scatter(meanList, stddevList, color = plotColours)
    plt.scatter([meanList[minIndex]], [stddevList[minIndex]], color = [defaultColours[len(optionTypes)]])
    for i in range(len(optionsList)):
        plt.annotate(optionsList[i], (meanList[i], stddevList[i]))
    plt.xlabel("Mitjana")
    plt.ylabel("Desviació estàndar")
    
    plt.subplot(212)
    plt.title("Temps d'execució")
    plt.bar(optionsList, execTimeList, color = plotColours)
    plt.xlabel("Opció")
    plt.ylabel("Temps(s)")
    plt.xticks(rotation=45)
    
    plt.show()


def plotBestKAccuracyExecTime(KList, accuracyLists, execTimeLists, optionsList, title = "Exactitud i temps d'execució mig de KMeans"):
    xTitle = "K"
    plotAccuracyExecTime(title, KList, xTitle, accuracyLists, execTimeLists, optionsList)

def plotBestKAccuracyExactExecTime(KList, accuracyListsExact, accuracyLists, execTimeLists, optionsList, title = "Exactitud i temps d'execució mig de KMeans"):
    xTitle = "K"
    plotAccuracyExactExecTime(title, KList, xTitle, accuracyListsExact, accuracyLists, execTimeLists, optionsList, showDots = False)
    
    
def plotFitToleranceAccuracyExecTime(toleranceList, accuracyLists, execTimeLists):
    title = "Exactitud i temps d'execució mig de KMeans"
    xTitle = "Tolerància a fit()"
    plotAccuracyExecTime(title, toleranceList, xTitle, accuracyLists, execTimeLists, None)

def plotFitToleranceAccuracyExactExecTime(KList, accuracyListsExact, accuracyLists, execTimeLists, optionsList, title = "Exactitud i temps d'execució mig de KMeans"):
    xTitle = "tolerància"
    plotAccuracyExactExecTime(title, KList, xTitle, accuracyListsExact, accuracyLists, execTimeLists, optionsList, showDots = False) 


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


def gatherDataGraph1():
    directory = "./graphData/graph1"
    fileList = listdir(directory)
    resultsList = []
    for fileName in fileList:
        fileNameSplit = fileName[:-4].split("_")[1:]
        K = int(fileNameSplit[-1][1:])
        initCentroidsMethod = "_".join(fileNameSplit[:-1])
        with open(directory + "/" + fileName, "r") as f:
            lines = f.readlines()
            lines = [float(i.replace("\n", "")) for i in lines]
            resultsList.append([initCentroidsMethod, K, lines[:-1], lines[-1]])
    
    KList = list(set([i[1] for i in resultsList]))
    KList.sort()
    
    optionsList = list(set([i[0] for i in resultsList]))
    optionsList.sort()
    
    accuracyListsExact = [[0 for _ in KList] for _ in optionsList]
    accuracyLists = [[0 for _ in KList] for _ in optionsList]
    
    execTimeLists = [[0 for _ in KList] for _ in optionsList]
    
    for el in resultsList:
        accuracyListsExact[optionsList.index(el[0])][el[1] - 1] = el[2][0]
        accuracyLists[optionsList.index(el[0])][el[1] - 1] = el[2][1]
        execTimeLists[optionsList.index(el[0])][el[1] - 1] = el[3]
    
    return KList, accuracyListsExact, accuracyLists, execTimeLists, optionsList

def generateGraph1():
    KList, accuracyListsExact, accuracyLists, execTimeLists, optionsList = gatherDataGraph1()
    # plotBestKAccuracyExecTime(KList, accuracyListsExact, execTimeLists, optionsList, title = "Exactitud (exacte) i temps d'execució mig de KMeans")
    # plotBestKAccuracyExecTime(KList, accuracyLists, execTimeLists, optionsList, title = "Exactitud (aproximada) i temps d'execució mig de KMeans")
    plotBestKAccuracyExactExecTime(KList, accuracyListsExact, accuracyLists, execTimeLists, optionsList)
    

def gatherDataGraph2():
    directory = "./graphData/graph2"
    fileList = listdir(directory)
    resultsList = []
    for fileName in fileList:
        fileNameSplit = fileName[:-4].split("_")[1:]
        tolerances = int(fileNameSplit[-1][3:])
        initCentroidsMethod = "_".join(fileNameSplit[:-1])
        with open(directory + "/" + fileName, "r") as f:
            lines = f.readlines()
            lines = [float(i.replace("\n", "")) for i in lines]
            resultsList.append([initCentroidsMethod, tolerances, lines[:-1], lines[-1]])
    
    toleranceList = list(set([i[1] for i in resultsList]))
    toleranceList.sort()
    
    optionsList = list(set([i[0] for i in resultsList]))
    optionsList.sort()
    
    accuracyListsExact = [[0 for _ in toleranceList] for _ in optionsList]
    accuracyLists = [[0 for _ in toleranceList] for _ in optionsList]
    
    execTimeLists = [[0 for _ in toleranceList] for _ in optionsList]
    
    for el in resultsList:
        print(el)
        accuracyListsExact[optionsList.index(el[0])][int(el[1] / 3)] = el[2][0]
        accuracyLists[optionsList.index(el[0])][int(el[1] / 3)] = el[2][1]
        execTimeLists[optionsList.index(el[0])][int(el[1] / 3)] = el[3]
    
    return toleranceList, accuracyListsExact, accuracyLists, execTimeLists, optionsList

def generateGraph2():
    toleranceList, accuracyListsExact, accuracyLists, execTimeLists, optionsList = gatherDataGraph2()
    # plotBestKAccuracyExecTime(KList, accuracyListsExact, execTimeLists, optionsList, title = "Exactitud (exacte) i temps d'execució mig de KMeans")
    # plotBestKAccuracyExecTime(KList, accuracyLists, execTimeLists, optionsList, title = "Exactitud (aproximada) i temps d'execució mig de KMeans")
    plotFitToleranceAccuracyExactExecTime(toleranceList, accuracyListsExact, accuracyLists, execTimeLists, optionsList)
 


def gatherDataGraph3():
    directory = "./graphData/graph3"
    fileList = listdir(directory)
    resultsList = []
    for fileName in fileList:
        fileNameSplit = fileName[:-4].split("_")[1:]
        tolerances = float(fileNameSplit[-1][3:])
        discriminant = "_".join(fileNameSplit[:-1])
        with open(directory + "/" + fileName, "r") as f:
            lines = f.readlines()
            lines = [float(i.replace("\n", "")) for i in lines]
            resultsList.append([(discriminant + ", " + str(tolerances)), lines[:-1], lines[-1]])
    
    optionsList = list(set([i[0] for i in resultsList]))
    optionsList.sort()
    
    meanList = [0 for _ in optionsList]
    stddevList = [0 for _ in optionsList]
    execTimeList = [0 for _ in optionsList]
    
    for el in resultsList:
        i = optionsList.index(el[0])
        meanList[i] = el[1][0]
        stddevList[i] = el[1][1]
        execTimeList[i] = el[2]
    
    return meanList, stddevList, execTimeList, optionsList

def generateGraph3():
    meanList, stddevList, execTimeList, optionsList = gatherDataGraph3()
    plotMeanStdDevExecTime("test", meanList, stddevList, execTimeList, optionsList)

def gatherDataGraph4(qAllowed = None):
    directory = "./graphData/graph4"
    fileList = listdir(directory)
    resultsList = []
    for fileName in fileList:
        fileNameSplit = fileName[:-4].split("_")[1:]
        K = int(fileNameSplit[0][1:])
        q = float(fileNameSplit[1][1:])
        readCond = (qAllowed == None)
        if not readCond:
            readCond = (q in qAllowed)
        if readCond:
            with open(directory + "/" + fileName, "r") as f:
                lines = f.readlines()
                lines = [float(i.replace("\n", "")) for i in lines]
                resultsList.append([K, str("q=" + str(q)), lines[0], lines[1]])

    KList = list(set([i[0] for i in resultsList]))
    KList.sort()
    
    
    optionsList = list(set([i[1] for i  in resultsList]))
    optionsList.sort()
    
    accuracyLists = [[0 for _ in KList] for _ in optionsList]
    execTimeLists = [[0 for _ in KList] for _ in optionsList]
    
    for el in resultsList:
        accuracyLists[optionsList.index(el[1])][el[0] - 1] = el[2]
        execTimeLists[optionsList.index(el[1])][el[0] - 1] = el[3]
        
    return KList, accuracyLists, execTimeLists, optionsList

def generateGraph4(qAllowed = None):
    KList, accuracyLists, execTimeLists, optionsList = gatherDataGraph4(qAllowed)
    plotKNNAccuracyExecTime(KList, accuracyLists, execTimeLists, optionsList)
    

if __name__ == '__main__':
    # generateTestPlots()
    # generateGraph1()
    # generateGraph2()
    # generateGraph3()
    generateGraph4([0.5, 1, 1.2, 1.4, 1.6, 1.8, 2, 3, 4, 5])
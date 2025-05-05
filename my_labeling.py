__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

from utils_data import read_dataset, read_extended_dataset, crop_images
from utils import get_color_prob, colors
from Kmeans import *
from KNN import *

import time
import matplotlib.pyplot as plt
import json

from collections import Counter

from CollectedData import *

if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here
    
    def Retrieval_by_color(imageList, labels, colours):
        validImages = list()
        for img, imgLabels in zip(imageList, labels):
            valid = True
            for colour in colours:
                valid = valid and (colour in imgLabels)
            if valid:
                validImages.append(img)
        return validImages
    
    def Retrieval_by_shape(imageList, labels, shapes):
        validImages = list()
        for img, imgLabels in zip(imageList, labels):
            valid = True
            for shape in shapes:
                valid = valid and (shape in imgLabels)
            if valid:
                validImages.append(img)
        return validImages
    
    def Retrieval_combined(imageList, colourLabels, shapeLabels, colours, shapes):
        validImages = list()
        for img, imgColourLabels, imgShapeLabels in zip(imageList, colourLabels, shapeLabels):
            valid = True
            for colour in colours:
                valid = valid and (colour in imgColourLabels)
            for shape in shapes:
                valid = valid and (shape in imgShapeLabels)
            if valid:
                validImages.append(img)
        return validImages
    
    
    def plotBars(title, xData, yData, xLabel, yLabel):
        plt.bar(xData, yData)
        plt.title(title)
        plt.xlabel(xLabel)
        plt.ylabel(yLabel)
        plt.show()
    
    def Kmean_statistics(kmeans, kMax):
        k = 2
        results = dict()
        while k <= kMax:
            print("Doing " + str(k))
            startTime = time.time()
            kmeans.K = k
            kmeans.fit()
            kmeans.withinClassDistance()
            endTime = time.time()
            resultVal = dict()
            resultVal["WCD"] = kmeans.WCD
            resultVal["iterations"] = kmeans.iteraciones
            resultVal["time"] = endTime - startTime
            results[k] = resultVal
            k += 1
        plotBars("Within-Class Distance (WCD)", list(range(2, kMax + 1)), list(results[i]["WCD"] for i in range(2, kMax + 1)), "K", "WCD")
        plotBars("Number of iterations", list(range(2, kMax + 1)), list(results[i]["iterations"] for i in range(2, kMax + 1)), "K", "Iterations")
        plotBars("Execution time", list(range(2, kMax + 1)), list(results[i]["time"] for i in range(2, kMax + 1)), "K", "Time (sec)")
        
    def Get_shape_accuracy(shapeLabels, groundTruth):
        correctCalculations = 0
        for calculatedLabel, correctLabel in zip(shapeLabels, groundTruth):
            # print(calculatedLabel)
            lst = calculatedLabel.tolist()
            if max(set(lst), key=lst.count) == correctLabel:
                correctCalculations += 1
        return correctCalculations / len(shapeLabels)
    
    def get_color_accuracy(colourLabels, groundTruth):
        correctCalculations = 0
        for calculatedLabel, correctLabel in zip(colourLabels, groundTruth):
            incorrectColours = 0
            for colour in calculatedLabel:
                if colour not in correctLabel:
                    incorrectColours += 1
            for colour in correctLabel:
                if colour not in calculatedLabel:
                    incorrectColours += 1
            totalColoursCombined = len(set(calculatedLabel + correctLabel))
            correctCalculations += (totalColoursCombined - incorrectColours) / totalColoursCombined
        return correctCalculations / len(colourLabels)
           
    
    def getBestKData(imageList):
        maxN = min(1000, len(imageList))
        print("Number of images: " + str(maxN))
        bestKList = list()
        for i in range(maxN):
            print("Checking image " + str(i))
            km = KMeans(imageList[i], 1, {"tolerance" : 5})
            km.find_bestK(12)
            bestKList.append(km.best_K)
        with open("results.txt", "a") as f:
            f.write(str(bestKList))
    
    def getBestMinkowskyQ(trainImgsList, trainClassLabels, testImgsList, testClassLabels):
        knn = KNN(trainImgsListmgs, train_ClassLabels)
        test_imgs_sample = np.random.choice(testImgsList)
        xData = []
        yData = []
        for q in range(1, 11):
            print("Trying q=" + str(q / 2))
            knn.get_k_neighbours(testImgsList, 5, q / 2)
            xData.append(q / 2)
            yData.append(Get_shape_accuracy(knn.neighbors, testClassLabels))
            print(str(q / 2) + " " + str(yData[-1]))
        plotBars("Accuracy of KNN", xData, yData, "q", "Accuracy")
        
    def testBestKAccuracy(trainImgsList, trainColourLabels):
        maxK = 9
        labels = {}
        for K in range(2, maxK + 1):
            labels[K] = []
        for i in range(len(trainImgsList)):
            img = trainImgsList[i]
            print("Testing image " + str(i))
            for K in range(2, maxK + 1):
                km = KMeans(img, K, {"tolerance" : K * 2})
                km.fit()
                labels[K].append(get_colors(km.centroids))
        accuracy = {}
        for K in labels.keys():
            accuracy[K] = get_color_accuracy(labels[K], trainColourLabels)
        return accuracy
    
    def testBestQforKNN(trainImgsList, trainClassLabels, testImgsList, testClassLabels):
        knn = KNN(trainImgsList, trainClassLabels)
        results = {}
        for q in range(1, 11):
            qReal = q / 2
            print("Trying q=" + str(qReal))
            knn.get_k_neighbours(testImgsList, 5, qReal)
            results[qReal] = Get_shape_accuracy(knn.neighbors, testClassLabels)
            print(str(qReal) + " " + str(results[qReal]))
        return results
    
    def testBestKforKNN(trainImgsList, trainClassLabels, testImgsList, testClassLabels):
        knn = KNN(trainImgsList, trainClassLabels)
        results = {}
        for K in range(1, 11):
            print("Trying K=" + str(K))
            knn.get_k_neighbours(testImgsList, K, 1.4)
            results[K] = Get_shape_accuracy(knn.neighbors, testClassLabels)
            print(str(K) + " " + str(results[K]))
        return results
    
    def getColourCentroids():
        arrIn = np.array([[0, 0, 0]])
        matOut = []
        for i in range(256):
            for j in range(256):
                print(str(i) + ", " + str(j))
                for k in range(0, 256, 8):
                    colourProbs = get_color_prob(np.array([[i, j, k], [i, j, k + 1], [i, j, k + 2], [i, j, k + 3], [i, j, k + 4], [i, j, k + 5], [i, j, k + 6], [i, j, k + 7]])).tolist()
                    for n in range(8):
                        matOut.append(colourProbs[n])
            print("Writing to file...")
            with open("colourProbs/return" + str(i) + ".txt", "a") as f:
                f.write(str(matOut) + "\n")
            matOut = []
        
        nColours = len(colors)
        probsList = []
        for i in range(256):
            outList = list([] for _ in range(nColours))
            print(i)
            with open("colourProbs/return" + str(i) + ".txt") as f:
                probsList = json.loads(f.read())
            for j in range(256):
                for k in range(256):
                    valid = -1
                    for l in range(len(probsList[j * 255 + k])):
                        if probsList[j * 255 + k][l] > 0.99:
                            valid = int(l)
                    if valid >= 0:
                        outList[valid].append([i, j, k])
            with open("colourProbs/validCoords" + str(i) + ".txt", "a") as f:
                f.write(str(outList) + "\n")
        
        nColours = len(colors)
        coordsList = list([] for _ in range(nColours))
        for i in range(256):
            print(i)
            with open("colourProbs/validCoords" + str(i) + ".txt") as f:
                newList = json.loads(f.read())
                for allCoords, newCoords in zip(coordsList, newList):
                    allCoords.extend(newCoords)
        avgCoords = []
        for coords in coordsList:
            totalSum = [0, 0, 0]
            for coord in coords:
                for i in range(len(coord)):
                    totalSum[i] += coord[i]
            for i in range(len(coord)):
                totalSum[i] /= len(coords)
            avgCoords.append(totalSum)
    
        print(avgCoords)
    
    """
    results = {}
    initOptions = ["custom"]
    for option in initOptions:
        results[option] = []
    for option in initOptions:
        print(option)
        i = 0
        for image in train_imgs[:100]:
            print(i)
            km = KMeans(image, 5, {"km_init" : "custom", "custom_option" : "center"})
            km.fit()
            results[option].append(get_colors(km.centroids))
            i += 1
            
    for option in results.keys():
        print(option + str(get_color_accuracy(results[option], train_color_labels[:100])))
    """
    
    def bestKAccuracy(testImgs, groundTruth):
        dev = 0
        i = 0
        n = 50
        for img, truth in zip(testImgs, groundTruth):
            km = KMeans(img, options = {"km_init" : "custom", "custom_option" : "fixed_centroids"})
            km.find_bestK(10, 0.392)
            print(str(i) + " " + str(len(truth)) + "  " + str(km.K))
            dev += len(truth) - km.K
            i += 1
        print(dev / n)
        
    bestKAccuracy(train_imgs[:50], train_color_labels[:50])
            
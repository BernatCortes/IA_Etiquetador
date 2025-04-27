__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

"""
Average best K:     5.033333333333333
Standard deviation: 1.2367736003600198
Minimum:            2
Maximum:            9
"""

from utils_data import read_dataset, read_extended_dataset, crop_images
from Kmeans import *
from KNN import *

import time
import matplotlib.pyplot as plt

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
        return correctCalculations
           
    
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
        for K in range(1, 51):
            print("Trying K=" + str(K))
            knn.get_k_neighbours(testImgsList, K)
            results[K] = Get_shape_accuracy(knn.neighbors, testClassLabels)
            print(str(K) + " " + str(results[K]))
        return results
    
    percentages = {}
    for K in accuracyKMeans_K.keys():
        percentages[K] = accuracyKMeans_K[K] / 1000
    print(percentages)
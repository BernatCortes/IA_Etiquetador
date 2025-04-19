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
            if calculatedLabel == correctLabel:
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
           
    
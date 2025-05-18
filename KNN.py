__authors__ = '[1672633, 1673893, 1673377]'
__group__ = '33'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        train_data = train_data.astype(float)
        self.train_data = train_data.reshape((train_data.shape[0], -1))

    def get_k_neighbours(self, test_data, k, q = 2):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_data = test_data.reshape((test_data.shape[0], -1))
        d = self.get_neighbour_distances(test_data, q)
        idx = np.argsort(d, axis=1)[:, :k]
        self.neighbors = self.labels[idx]
        # return d
        
    def get_neighbour_distances(self, test_data, q = 2):
        if q == 2:
            return cdist(test_data, self.train_data)
        elif q == 1:
            return cdist(test_data, self.train_data, 'cityblock')
        else:
            return cdist(test_data, self.train_data, 'minkowski', p=q)
        

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        test_class = []

        for neighbors in self.neighbors:
            values, cantidad = np.unique(neighbors, return_counts=True)

            max_cantidad = np.max(cantidad)
            candidatos = values[cantidad == max_cantidad]

            if len(candidatos) > 1:
                for label in neighbors:
                    if label in candidatos:
                        test_class.append(label)
                        break
            else:
                test_class.append(candidatos[0])

        return np.array(test_class)

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()

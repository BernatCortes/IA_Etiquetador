__authors__ = '[1672633, 1673893, 1673377]'
__group__ = '33'

import numpy as np
import utils
from random import sample

# Posicions de centroides calculades per a cada color
colourCentroids = [
    [172.34569607498716, 31.588923719554803, 77.83681296833197 ], 
    [226.43081195991644, 119.70810098347359, 135.86842861962992], 
    [119.14937212194089, 61.648907533170274, 82.29653744722135 ], 
    [227.30506395771258, 199.12416706105589, 134.32113504109424], 
    [88.7354416660608  , 190.4386538750928 , 126.16016027228873], 
    [62.010482475756845, 139.8645820599956 , 131.50433378250696], 
    [129.87274513542542, 47.93495600915736 , 132.27861837004775], 
    [222.46513886603404, 80.86385899475253 , 130.76395736200246], 
    [23.875163019852195, 22.996087523547313, 46.021880886827994], 
    [123.07999646424467, 123.46574737028197, 130.38557411827102], 
    [243.63584238135903, 247.27558862998657, 233.1754300353788 ]
]

class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options
        self.labels = None
        self.old_centroids = None
        self.WCD = 1
        self.best_K= None

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        X = np.asarray(X, dtype=np.float64) 
        if X.ndim == 3 and X.shape[2] == 3:
            F,C, _ = X.shape
            X = X.reshape(F*C,3)
        elif X.ndim != 2:
            print("Error")
        self.X = X

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        N, D = self.X.shape
        self.centroids = np.zeros((self.K, D))

        if self.options['km_init'].lower() == 'first':
            _, indicesU = np.unique(self.X, axis=0, return_index=True)
            puntosU = self.X[np.sort(indicesU)]
            self.centroids = puntosU[:self.K]

        elif self.options['km_init'].lower() == 'random':
            indices = np.random.choice(N, self.K, replace=False)
            self.centroids = self.X[indices]
        
        elif self.options['km_init'].lower() == 'custom':
            if self.options['custom_option'].lower() == "random_distance":
                minDistance = ((255 ** (3 / 2)) / self.K) * int(self.options['random_distance_d'].lower())
                while len(self.centroids) < self.K:
                    newCentroid = np.ndarray([np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)])
                    minDist = 255 ** (3 / 2)
                    for centroid in self.centroids:
                        minDist = min(minDist, np.linalg.norm(newCentroid - centroid))
                    if minDist < minDistance:
                        self.centroids.append(newCentroid)
            
            elif self.options['custom_option'].lower() == "fixed_centroids":
                self.centroids = np.array(sample(colourCentroids, self.K))
                
            elif self.options['custom_option'].lower() == "const_X_dist":
                if self.K == 1:
                    self.centroids = np.array(self.X[np.floor(len(self.X) / 2)])
                else:
                    self.centroids = []
                    dist = len(self.X) / (self.K - 1)
                    for i in range(self.K - 1):
                        self.centroids.append(self.X[np.floor(i * dist)])
                    self.centroids.append(self.X[-1])
                    self.centroids = np.array(self.centroids)
                    
            elif self.options['custom_option'].lower() == "center":
                center = int(np.floor((len(self.X) - self.K) / 2))
                self.centroids = []
                for i in range(center, center + int(self.K)):
                    self.centroids.append(self.X[i])
                self.centroids = np.array(self.centroids)
            

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        dist = distance(self.X, self.centroids)
        self.labels = np.argmin(dist, axis=1)
        
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.old_centroids = np.copy(self.centroids)
        
        centroidsN = np.zeros_like(self.centroids)
        
        for k in range(self.K):
            puntosCentroid = self.X[self.labels == k]
        
            if len(puntosCentroid) > 0:
                centroidsN[k] = np.mean(puntosCentroid, axis=0)
            else:
                centroidsN[k] = self.centroids[k]
                
        self.centroids = centroidsN
        """
        
        # Save the old centroids before updating them
        self.old_centroids = np.copy(self.centroids)

        # Initialize an array to hold the new centroids
        centroidsN = np.zeros_like(self.centroids)
    
        # Use np.add.at to accumulate the sum of points assigned to each centroid
        np.add.at(centroidsN, self.labels, self.X)
    
        # Count how many points are assigned to each centroid (avoid division by zero)
        counts = np.bincount(self.labels, minlength=self.K)
    
        # Divide the sum of points by the number of points for each centroid
        # where the count is greater than zero (to avoid division by zero)
        for k in range(self.K):
            if counts[k] > 0:
                centroidsN[k] /= counts[k]
            else:
                centroidsN[k] = self.centroids[k]  # Keep the old centroid if no points are assigned
    
        # Update the centroids
        self.centroids = centroidsN
        """

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        # print(np.linalg.norm(self.centroids - self.old_centroids))

        if np.linalg.norm(self.centroids - self.old_centroids) > self.options['tolerance']:
            return False

        return True 

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        P, D = self.X.shape

        self._init_centroids()
        self.iteraciones = 0 

        while self.iteraciones < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            self.iteraciones += 1

            if self.converges():
                break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        N = self.X.shape[0]
        total_distance = 0

        for i in range(N):
            x = self.X[i]
            cluster_idx = self.labels[i]
            Cx = self.centroids[cluster_idx]

            distancia_cuadrada = np.sum((x - Cx) ** 2)

            total_distance += distancia_cuadrada

        self.WCD = total_distance / N
        
    def interClassDistance(self):
        distanceSum = 0
        for c1 in self.centroids:
            for c2 in self.centroids:
                distanceSquared = 0
                for d1, d2 in zip(c1, c2):
                    distanceSquared += np.square(d1 - d2)
                distanceSum += np.sqrt(distanceSquared)
        self.ICD = distanceSum / (np.square(len(self.centroids)) - len(self.centroids))
        
    def fisherDiscriminant(self):
        self.withinClassDistance()
        self.interClassDistance()
        self.fisher = self.WCD / self.ICD
                

    def find_bestK(self, max_K, tolerance = 0.2):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.K = 1
        self.fit()
        self.withinClassDistance()
        anteriorWCD = self.WCD
        self.K += 1
        
        millorKTrobada = False
        
        while not millorKTrobada:
            self.fit()
            self.withinClassDistance()
            millorKTrobada = ((1 - (self.WCD / anteriorWCD)) < tolerance) or (self.K == max_K)
            if not millorKTrobada:
                self.K += 1
                anteriorWCD = self.WCD
                
        self.K -= 1
        self.best_K = self.K
        

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    P = X.shape[0]
    K = C.shape[0] 

    dist = np.zeros((P, K))

    for i in range(P):
        for j in range(K):
            diff = X[i] - C[j] 
            dist[i, j] = np.sqrt(np.sum(diff ** 2))

    return dist


def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    probabilidadC = utils.get_color_prob(centroids)
    colores = utils.colors
    color_indices = np.argmax(probabilidadC, axis=1)

    return [colores[i] for i in color_indices]

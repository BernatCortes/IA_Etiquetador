__authors__ = '[1672633, 1673893, 1673377]'
__group__ = '33'

import numpy as np
import utils


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
            pass
            

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

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

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
        iteraciones = 0 

        while iteraciones < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            iteraciones += 1

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

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        self.K = 1
        self.fit()
        anteriorWCD = self.withinClassDistance()
        self.K += 1
        
        millorKTrobada = False
        
        while not millorKTrobada:
            self.fit()
            actualWCD = self.withinClassDistance()
            millorKTrobada = ((actualWCD / anteriorWCD) < 0.2)
            if not millorKTrobada:
                self.K += 1
        

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
    return list(utils.colors)

import numpy as np
import matplotlib.pyplot as plt
import scipy
import random
from sklearn.datasets import load_iris

def kMeans2D(data, k, maxIterations=100):
    """
    Performs K Means clustering based on an input of a 2D dataset.
    
    Parameters
    ----------
    data:           tuple, shape (xValues, yValues)
                    Which data to create clusters from
    k:              int
                    Amount of clusters to create
    maxIterations:  int
                    Amount of maximum iterations to perform
    Returns
    -------
    kData:          ndarray, shape (data, k)
                    The categorized data, split into K clusters.
    """
    # Select K random points in the dataset
    random.seed()
    clusters = random.sample(range(np.shape(data)[0]), k)

    # Print the selected indexes from data
    for i in range(k):
        print("Point", i + 1, "index:", clusters[i])

    # Find the shortest distance between each cluster and unassigned point
    # By using the distance cluster.x + cluster.y - (point.x + point.y)
    # And assign the point to the cluster with the least distance
    #data = np.expand_dims(data, axis=2)
    converged = False
    iterations = 1
    mean = []

    while not converged and iterations < maxIterations:
        # Create an array as large as the N amount of samples in the data
        # To store the selected cluster for each sample
        clusterIndexes = np.zeros(np.shape(data)[0])

        for i in range(np.shape(data)[0]):
            distances = []                      # List of K distances

            for j in range(k):                  # Find the distance of each cluster J to the current point I
                clusterIndex = clusters[j]

                # First iteration
                if iterations == 1:             
                    clusterVal = (data[clusterIndex])[0] + (data[clusterIndex])[1]
                    clusterVal = np.linalg.norm()
                    pointVal = ((data[i])[0] + (data[i])[1])

                # Not the first iteration, use mean values
                else:                          
                    clusterVal = mean[j]
                    pointVal = ((data[i])[0] + (data[i])[1])

                # Calculate the values and store it in the distances list
                dist = np.abs(clusterVal - pointVal)
                distances.append(dist)

            # Finds the smallest distance and its index from the distances list
            print(distances)
            value, index = min((value, index) for (index, value) in enumerate(distances))
            print("Smallest distance of", value, "with cluster", index, '\n')

            # Adds the index for datapoint #i to clusterIndexes
            clusterIndexes[i] = index

        print("Selected clusters:", clusterIndexes)

        # Calculate the mean of each cluster
        oldMean = mean.copy()
        newMean = []
        for i in range(k):
            newMean.append(np.mean((data)[clusterIndexes == i]))

        if oldMean == newMean:
            converged = True

        if iterations == 1:
            converged = True

        else:
            print("New mean:", newMean)
            oldMean = newMean
            mean = newMean.copy()

        iterations += 1

    #return kData


# Load the IRIS dataset
iris = load_iris()

# Prints some information about the set
print ("Iris species: ", iris.target_names)
print ("Iris features: ", iris.feature_names)
print ("Data: ", iris.data[0:4])
print ("Labels: ", iris.target)
print ("Shape of data:", np.shape(iris.data))

# Separates the data (select only sepal widths/lengths)
data = iris.data[:, :2]

irisClusters = []

numIterations = 1                      # Number of iterations of the algorithm
k = 3                                  # Create 3 clusters (since there are 3 different flowers)

for i in range(numIterations):
    irisClusters.append(kMeans2D(data, 3))  # Perform kMeansClustering, append the resulting clusters to list of clusters

#print(irisClusters)
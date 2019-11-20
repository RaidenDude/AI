import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
import scipy
from sklearn.datasets import load_sample_images
from sklearn.datasets import load_digits
import random

def manualPCA(image, subset=0.9):
    """
    Manually performs PCA on an image by normalizing the data, creating a co-variance matrix, 
    calculating eigenvalues/vectors from the co-variance matrix, sorting the eigenvalues/vectors based on decreasing significance,
    creates a matrix of the 'image.size * subset' first number of vectors, 
    multiplies this matrix with the co-variance matrix to create a matrix Z,
    multiplies Z with Z transposed, adds back the mean which was subtracted earlier for normalization,
    and returns the matrix after this calculation.

    Parameters
    ----------
    image:  ndarray, shape (x_resolution, y_resolution)
            One of (or all of, if the image is black/white) the channels from the starting image from imread.
    subset: np.float64 from [0.1, 0.99]
            Contains the factor of eigenvalues to use for creation of the Z matrix
    
    Returns
    -------
    Z:      ndarray, shape (x_resolution, y_resolution)
            The Z matrix (image channel) after all calculation/compression is completed.
    """

    # Normalize the single channel by subtracting the mean of that channel
    mean = np.mean(image)
    covMatrix = image - mean

    # Now you may treat the normalized single channel image as the co-variance matrix for PCA
    eigenValues, eigenVectors = np.linalg.eigh(np.cov(covMatrix))

    # Sort the eigenvalues and the corresponding eigenvectors
    index = eigenValues.argsort()[::-1]   
    eigenValues = eigenValues[index]
    eigenVectors = eigenVectors[:, index]

    # Choose the number of eigenvectors for the reconstruction and organize them in a matrix
    n = image[0].size
    L = int (n * subset)         # Select a subset of the eigenvectors as basis vectors
    W = np.zeros(L)

    W = eigenVectors[:, :L]     # Copies the L first values from eigenVectors to W

    # Multiply eigenvector matrix with the co-variance matrix, call the product Z
    Z = np.dot(W.T, covMatrix)

    # Now multiply Z with Z transposed to correct the order of the matrix
    Z = np.dot(W, Z)

    # Add the mean that we subtracted in the beginning for normalization purposes
    Z += mean

    return Z


# USING SKLEARN DATASETS:
# Sample_images (flower/china)
dataset = load_sample_images()
print("Amount of images:", len(dataset.images))

image1 = dataset.images[0]
print("Shape of dataset:", np.shape(image1))

plt.imshow(image1)
plt.title("Original image")
plt.show()

# Crops the photo so it is square (x*x resolution instead of x*y, where x != y)
# First selects the smallest of the axis as the resolution to use for both x and y
if np.size(image1, 0) > np.size(image1, 1):
    minRes = np.size(image1, 1)

else:
    minRes = np.size(image1, 0)

# Then selects only the 'minRes' first pixels of each axis
image1 = image1[:minRes, :minRes]

# Performs manual PCA on the image
image1[:, :, 0] = manualPCA(image1[:, :, 0])
image1[:, :, 1] = manualPCA(image1[:, :, 1])
image1[:, :, 2] = manualPCA(image1[:, :, 2])

# Displays the new image + prints the shape (xRes, yRes, channels (RGB)) and min/max values
plt.imshow(image1)
plt.title("Compressed image")
plt.show()

# DIGITS:
# Load_digits (0-9 handwritten numbers)
digits = load_digits()

n = digits.data.shape[0]
print("Amount of images:", n)

image1 = digits.images[0]
print("Shape of dataset:", np.shape(image1))

#compressedDigits = np.zeros((np.shape(image1)[0], np.shape(image1)[0], n))
compressedDigits = []

for i in range(n):
    compressedDigits.append(manualPCA(digits.images[i]))

compressedDigits = np.asarray(compressedDigits)
print("Shape:", np.shape(compressedDigits))

random.seed()
index = random.randint(0, n)

# Original digit:
plt.imshow(digits.images[index], cmap="gray")
plt.title("Original image (digit #" + str(index) + " )")
plt.show()

# Compressed digit:
plt.imshow(compressedDigits[index], cmap="gray")
plt.title("Compressed image (digit #" + str(index) + " )")
plt.show()
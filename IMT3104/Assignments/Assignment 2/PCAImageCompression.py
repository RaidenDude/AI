import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage.io import imread
import scipy

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

# BLACK / WHITE IMAGE
# Reads black/white image
bwImage = imread("Assignment 2/BWImage.png", as_gray=True)

# Transforms the numbers from [0, 255] to [0.0, 1.0] since it's black/white
bwImage = bwImage / 255.0
bwImage[bwImage < 0.0] = 0.0
bwImage[bwImage > 1.0] = 1.0

# Displays the image + prints the shape (xRes, yRes) and min/max values
plt.imshow(bwImage, cmap="gray")
plt.title("Original black/white image")
plt.show()

print("Shape of original black/white image:", np.shape(bwImage))
print("Min value:", np.min(bwImage), "Max value:", np.max(bwImage), "\n")

# Performs manual PCA on the image (with 90% of eigenvectors)
bwImageCompressed = manualPCA(bwImage)

# Displays the new image + prints the shape (xRes, yRes) and min/max values
plt.imshow(bwImageCompressed, cmap="gray")
plt.title("Compressed black/white image, 90% of eigenvectors")
plt.show()

# Performs manual PCA on the image (with 20% of eigenvectors)
bwImageCompressed = manualPCA(bwImage, subset=0.2)

# Displays the new image + prints the shape (xRes, yRes) and min/max values
plt.imshow(bwImageCompressed, cmap="gray")
plt.title("Compressed black/white image, 20% of eigenvectors")
plt.show()

# Reads RGB image
rgbImage = imread("Assignment 2/RGBImage.png", as_gray=False)

# Displays the image + prints the shape (xRes, yRes, channels (RGB)) and min/max values
plt.imshow(rgbImage)
plt.title("Original RGB image")
plt.show()

print("Shape of original RGB image:", np.shape(rgbImage))
print("Min value:", np.min(rgbImage), "Max value:", np.max(rgbImage), "\n")

rgbImageCompressed = rgbImage
# Performs manual PCA on the image (with 90% of eigenvectors)
rgbImageCompressed[:, :, 0] = manualPCA(rgbImage[:, :, 0])
rgbImageCompressed[:, :, 1] = manualPCA(rgbImage[:, :, 1])
rgbImageCompressed[:, :, 2] = manualPCA(rgbImage[:, :, 2])

# Displays the new image
plt.imshow(rgbImageCompressed)
plt.title("Compressed RGB image, 90% of eigenvectors")
plt.show()

rgbImageCompressed = rgbImage
# Performs manual PCA on the image (with 20% of eigenvectors)
rgbImageCompressed[:, :, 0] = manualPCA(rgbImage[:, :, 0], subset=0.2)
rgbImageCompressed[:, :, 1] = manualPCA(rgbImage[:, :, 1], subset=0.2)
rgbImageCompressed[:, :, 2] = manualPCA(rgbImage[:, :, 2], subset=0.2)

# Displays the new image
plt.imshow(rgbImageCompressed)
plt.title("Compressed RGB image, 20% of eigenvectors")
plt.show()
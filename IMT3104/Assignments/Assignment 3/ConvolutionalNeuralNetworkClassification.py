# Similar architecture as exercise 5
# Use CIFAR-10 dataset instead of MNIST
# Train/evaluate on:
# * Classical RGB-channels of an image as input to the network
# * Using a HLG-channel as input to the network, where:
#   H = HoG from original image,
#   L = LBP from original image
#   G = gray-level of the original image

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, local_binary_pattern
from keras.datasets import cifar10

def getHLG(image):
    """
    Arguments:
    image -- input image consisting of [R, G, B], shape (32, 32, 3) for CIFAR-10
    
    Returns:
    HLG   -- output consisting of [Histogram of Oriented Gradients, Local Binary Patterns, Graylevel], shape (32, 32, 3) for CIFAR-10
    """

    # First calculates the HOG:
    featureData, HOG = hog(image, visualize=True, multichannel=True)

    # Then finds the graylevel using the ITU-R 601-2 luma transformation:
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    graylevel = R * 299/1000 + G * 587/1000 + B * 114/1000

    # Then calculates the LBP:
    LBP = local_binary_pattern(graylevel, 8, 1)

    # Constructs the resulting HLG image (32, 32, 3):
    HLG = np.zeros((32, 32, 3))
    HLG[:, :, 0] = HOG
    HLG[:, :, 1] = LBP
    HLG[:, :, 2] = graylevel

    return HLG

# Loads the CIFAR-10 dataset, and splits it into training/testing sets (X = data, Y = labels)
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

# Retrieves the HLG of the training set
HLG = getHLG(X_train[2])

# Prints the shapes of the different images/datasets
print("Shape of X_train: \t\t ", np.shape(X_train))
print("Shape of each X_train image (X[2]): \t", np.shape(X_train[2]))
print("Shape of HLG for each image (X[2]): \t", np.shape(HLG))

# Plots the different properties for the image
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 8), sharex=True, sharey=True)

# Original
ax1.axis('off')
ax1.imshow(X_train[2])
ax1.set_title("Original image")

# HOG
ax2.axis('off')
ax2.imshow(HLG[:, :, 0], cmap=plt.cm.gray)
ax2.set_title("Histogram of oriented gradients")

# LBP
ax3.axis('off')
ax3.imshow(HLG[:, :, 1], cmap=plt.cm.gray)
ax3.set_title("Local binary pattern")

# Gray
ax4.axis('off')
ax4.imshow(HLG[:, :, 2], cmap=plt.cm.gray)
ax4.set_title("Graylevel")

# Show the plots
plt.show()

# Convolutional Layer 1.
filter_size1 = 5          # Convolution filters are 5 x 5 pixels.
num_filters1 = 16         # There are 16 of these filters.

# Convolutional Layer 2.
filter_size2 = 5          # Convolution filters are 5 x 5 pixels.
num_filters2 = 36         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.


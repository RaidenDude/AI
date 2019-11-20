import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist                        # For the MNIST dataset
from sklearn import svm                                 # Support Vector Machines
from sklearn.ensemble import RandomForestClassifier     # Random Forest estimator
from sklearn.linear_model import LogisticRegression     # Logistic Regression model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import datasets

# Loads the MNIST dataset, containing images of numbers/digits with 28x28 pixels
digits = datasets.load_digits()

# Flattens the data into a (samples, features) matrix
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# RANDOM FOREST CLASSIFICATION:
# Creates a gaussian classifier
clf = RandomForestClassifier(n_estimators=10000)

# Uses training data from 80% of the dataset
training_samples = int(np.floor(n_samples * 0.8))
clf.fit(data[:training_samples], digits.target[:training_samples])

# Get the expected and predicted values for the rest of the dataset
expected = digits.target[training_samples:]
randomForest = clf.predict(data[training_samples:])

# Displays the score and classification report
print("Random forest classification accuracy: ", metrics.accuracy_score(expected, randomForest))
print(metrics.classification_report(expected, randomForest))

# Displays some of the numbers and their predictions
images_and_predictions = list(zip(digits.images[training_samples:], randomForest))
for index, (image, randomForest) in enumerate(images_and_predictions[:4]):
    fig = plt.figure(0)
    fig.canvas.set_window_title('Random Forest predictions')
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % randomForest)
plt.show()

# TODO: Legge til SVM og Logistic Regression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris                  # For the iris dataset
from sklearn import svm                                 # Support Vector Machines
from sklearn.ensemble import RandomForestClassifier     # Random Forest estimator
from sklearn.linear_model import LogisticRegression     # Logistic Regression model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Loads the iris dataset, containing three different flower species with different average sepal/petal widths/lengths
iris = load_iris()

# Prints some information about the set
print ("Iris species: ", iris.target_names)
print ("Iris features: ", iris.feature_names)
print ("Data: ", iris.data[0:4])
print ("Labels: ", iris.target)

# Separates the data (sepal/petal widths/lengths) to X, and labels to y
X = iris.data[:, :2]
y = iris.target

# Split the dataset into training and testing sets, 20% reserved for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Plot the training points, with sepal attributes
# 
plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Set1, edgecolor='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

# RANDOM FOREST CLASSIFICATION:
# Creates a gaussian classifier
clf = RandomForestClassifier(n_estimators=10000)

# Trains the model
clf.fit(X_train, y_train)
randomForest = clf.predict(X_test)

# Displays the score and classification report
print("Random forest classification accuracy: ", metrics.accuracy_score(y_test, randomForest))
print(metrics.classification_report(y_test, randomForest))

# SUPPORT VECTOR MACHINES:
# Creates a linear support vector classifier
clf = svm.LinearSVC(max_iter=10000)

# Trains the model
clf.fit(X_train, y_train)
linearSVC = clf.predict(X_test)

# Displays the score and classification report
print("Linear support vector classification accuracy: ", metrics.accuracy_score(y_test, linearSVC))
print(metrics.classification_report(y_test, linearSVC))

# Creates a non-linear support vector classifier
clf = svm.SVC(max_iter=10000, gamma='auto')

# Trains the model
clf.fit(X_train, y_train)
nonLinearSVC = clf.predict(X_test)

# Displays the score and classification report
print("Non-linear support vector classification accuracy: ", metrics.accuracy_score(y_test, nonLinearSVC))
print(metrics.classification_report(y_test, nonLinearSVC))

# LOGISTIC REGRESSION:
# Creates a logistic regression classifier
clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=10000)

clf.fit(X_train, y_train)
logReg = clf.predict(X_test)

# Displays the score and classification report
print("Logistic regression classification accuracy: ", metrics.accuracy_score(y_test, logReg))
print(metrics.classification_report(y_test, logReg))

# Uses PCA to transform the training dataset from 4 to 3 dimensions by dimentionality reduction
#fig = plt.figure(1, figsize=(8, 6))
#ax = Axes3D(fig, elev=-150, azim=110)
#X_reduced = PCA(n_components=3).fit_transform(X)
#ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=y,
#           cmap=plt.cm.Set1, edgecolor='k', s=40)
#ax.set_title("First three PCA directions")
#ax.set_xlabel("1st eigenvector")
#ax.w_xaxis.set_ticklabels([])
#ax.set_ylabel("2nd eigenvector")
#ax.w_yaxis.set_ticklabels([])
#ax.set_zlabel("3rd eigenvector")
#ax.w_zaxis.set_ticklabels([])
#plt.show()
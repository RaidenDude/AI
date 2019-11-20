#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


data = np.loadtxt(open("ex1data2.txt", "r"), delimiter=",")
X = data[:, 0:2] #Reading two features from the input file i.e. the area in squre feet and the number of bedrooms
y = data[:, 2]   #labels in the form of price of houses
m = len(y)

# Print out some data points
print ('First 3 examples from the dataset:\n',)
for i in range(3):
    print ('x = [area bedrooms ->]', X[i, ], ', y =', y[i])


# In[2]:


def feature_normalize(X):
    """
    Normalizes the features in x.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Features to be normalized.

    Returns
    -------
    X_norm : ndarray, shape (n_samples, n_features)
        A normalized version of X where the mean value of each feature is 0 and the standard deviation is 1.
    mu : ndarray, shape (n_features,)
        The mean value.
    sigma : ndarray, shape (n_features,)
        The standard deviation.
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma


# In[3]:


X, mu, sigma = feature_normalize(X)
X = np.hstack((np.ones((m, 1)), X))
#print(X)
print ('First 3 normalized examples from the dataset:\n',)
for i in range(3):
    print ('x = [Constant area bedrooms ->]', X[i, ], ', y =', y[i])


# In[ ]:


# Choose the learning rate alpha 
alpha = 0.1
num_iters = 50

# Init Beta and run gradient descent
beta = np.zeros(3) #(B_0, B_1, B_2)


# In[ ]:


def compute_cost_multi(X, y, beta):
    """
    Compute cost for linear regression with multiple variables.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    beta : ndarray, shape (n_features,)
        Linear regression parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using beta as the parameter for linear regression to fit the data points in X and y.
    """
    m = len(y)
    diff = X.dot(beta) - y
    J = 1.0 / (2 * m) * diff.T.dot(diff)
    return J


# In[ ]:


def gradient_descent_multi(X, y, beta, alpha, num_iters):
    """
    Performs gradient descent to learn beta.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    beta : ndarray, shape (n_features,)
        Initial linear regression parameter.
    alpha : float
        Learning rate.
    num_iters: int
        Number of iteration.

    Returns
    -------
    beta : ndarray, shape (n_features,)
        Linear regression parameter.
    J_history: ndarray, shape (num_iters,)
        Cost history.
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        beta -= alpha / m * ((X.dot(beta) - y).T.dot(X))
        J_history[i] = compute_cost_multi(X, y, beta)

    return beta, J_history


# In[ ]:


beta, J_history = gradient_descent_multi(X, y, beta, alpha, num_iters)


# In[ ]:


import matplotlib.pyplot as plt


plt.figure()
plt.plot(range(1, num_iters + 1), J_history, color='g')
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


# In[ ]:


print ('beta computed from gradient descent:')
print (beta)


# In[ ]:


#Training is complete. The model can be tested with test data. 


# In[ ]:


normalize_test_data = ((np.array([2650, 4]) - mu) / sigma)
normalize_test_data = np.hstack((np.ones(1), normalize_test_data))
price = normalize_test_data.dot(beta)
print ('Predicted price of a 2650 sq-ft house with 4 rooms:', price)


# In[ ]:


normalize_test_data = ((np.array([5650, 6]) - mu) / sigma)
normalize_test_data = np.hstack((np.ones(1), normalize_test_data))
price = normalize_test_data.dot(beta)
print ('Predicted price of a 5650 sq-ft house with 6 rooms:', price)


# In[ ]:





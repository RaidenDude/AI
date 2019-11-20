#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = np.loadtxt(open("F:\Code\AI\W36\ex1data1.txt", "r"), delimiter=",")
X = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
print("Input", X)
print("Output", y)
print("Number of Datapoints", m)


# In[3]:


def plot_data(x, y):
    """
    Plots the data points x and y.

    Parameters
    ----------
    x : array-like
        Data on x axis.
    y : array-like
        Data on y axis.
    """
    plt.plot(x, y, linestyle='', marker='*', color='c', label='Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')


# In[4]:


plt.figure()
plot_data(X, y)
plt.show()


# In[5]:


n = len(X)
print(n)
print(X.shape)


# In[6]:


X = np.hstack((np.ones((m, 1)), X.reshape(m, 1)))
print("Shape of X for matrix multiplication", X.shape)
print("Shape of Y", y.shape)


# In[7]:


theta = np.zeros(2)
print(theta)


# In[8]:


iterations = 20000
alpha = 0.01


# In[9]:


def compute_cost(x, y, theta):
    """
    Compute cost for linear regression.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Linear regression parameter.

    Returns
    -------
    J : numpy.float64
        The cost of using theta as the parameter for linear regression to fit the data points in X and y.
    """
    m = len(y)
    J = np.sum(np.square(x.dot(theta) - y)) / (2.0 * m)

    return J


# In[10]:


cost = compute_cost(X, y, theta)
print ('The cost on initial theta:', cost)


# In[11]:


def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn theta.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y : ndarray, shape (n_samples,)
        Labels.
    theta : ndarray, shape (n_features,)
        Initial linear regression parameter.
    alpha : float
        Learning rate.
    num_iters: int
        Number of iteration.

    Returns
    -------
    theta : ndarray, shape (n_features,)
        Linear regression parameter.
    J_history: ndarray, shape (num_iters,)
        Cost history.
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        theta -= alpha / m * ((X.dot(theta) - y).T.dot(X))
        print("theta", theta," at iteration i =", i)
        J_history[i] = compute_cost(X, y, theta)
        print("Iteration i= ",i,"Cost = ",J_history[i])

    return theta, J_history


# In[12]:


theta, _ = gradient_descent(X, y, theta, alpha, iterations)


# In[13]:


print ("Theta found by gradient descent: \n", theta)


# In[14]:


plt.figure()
plot_data(X[:, 1], y)
plt.plot(X[:, 1], X.dot(theta), label='Linear Regression')
plt.legend(loc='upper left', numpoints=1)
plt.show()


# In[16]:


predict1 = np.array([1, 3.5]).dot(theta)
print ("For population = 35,000, we predict a profit of", predict1 * 10000)

predict2 = np.array([1, 7]).dot(theta)
print ("For population = 70,000, we predict a profit of", predict2 * 10000)

predict3 = np.array([1, 9]).dot(theta)
print ("For population = 90,000, we predict a profit of", predict3 * 10000)

predict4 = np.array([1, 22]).dot(theta)
print ("For population = 220,000, we predict a profit of", predict4 * 10000)


# In[ ]:





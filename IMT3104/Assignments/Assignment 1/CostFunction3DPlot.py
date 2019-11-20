import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_cost(x, y, m):
    """
    Compute cost for linear regression.

    Parameters
    ----------
    x: ndarray, shape (n_samples, n_features)
        Training data, where n_samples is the number of samples and n_features is the number of features.
    y: ndarray, shape (n_samples)
        Labels.
    m: numpy.int32
       Size/length of the x/y np.arrays
    
    Returns
    -------
    numpy.float64
        The cost of the parameter for linear regression to fit the data points in x and y.
    """
    return np.sum(np.square(x - y)) / (2.0 * m)

data = np.loadtxt(open("F:\Code\AI\W36\ex1data1.txt", "r"), delimiter=",")
x = data[:, 0] # population of city
y = data[:, 1] # profit of food trucks
m = len(y)

cost = compute_cost(x, y, m)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x, y = np.meshgrid(x, y)
zs = np.array([compute_cost(xi, yi, m) 
               for xi, yi in zip(np.ravel(x), np.ravel(y))])
z = zs.reshape(y.shape)

customCmap = plt.get_cmap("gist_earth")
plot = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=customCmap, edgecolor='none')
fig.colorbar(plot, ax=ax, shrink=0.5, aspect=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('cost')

plt.show()
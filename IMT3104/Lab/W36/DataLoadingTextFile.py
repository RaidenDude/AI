#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = np.loadtxt(open("F:\Code\AI\W36\ex1data1.txt", "r"), delimiter=",")
x = data[:, 0]
y = data[:, 1]
m = len(y)  # Number of training examples
#print("Input", x)
#print("Output", y)
print("Number of Datapoints", m)


# In[6]:


plt.plot(x, y, linestyle='', marker='*', color='c', label='Training data')
plt.xlabel('Input x')
plt.ylabel('Label y')
plt.show()


# In[ ]:






# coding: utf-8

# In[285]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def iterate_map(f, x_0, n):
    # extended for multi-dim case
    # trajectory = [x_0] * (n+1)
    trajectory = np.stack([x_0] * (n+1), axis=0)
    trajectory[0] = x_0
    for i in range(1, n + 1):
        trajectory[i] = f(*trajectory[i-1])
    
    return trajectory


# In[286]:


henon = lambda a, b: lambda x, y: np.asarray((a - (x**2) + b * y, x))


# In[287]:


def problem_4(a):
    iterations = 500
    trajectory = iterate_map(henon(a, 0.4), np.asarray((0.8, 0.8)), iterations)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].set_ylabel('$x_n$')
    axes[1].set_ylabel('$y_n$')
    axes[1].set_xlabel('n')
    axes[0].scatter(np.arange(iterations + 1), trajectory[:,0], s=2)
    axes[1].scatter(np.arange(iterations + 1), trajectory[:,1], s=2)
    
    fig.suptitle('Trajectories for $a={0}$'.format(a), y=0.93)
    
    fig.show()
    
#     return trajectory


# In[288]:


problem_4(0.1)
problem_4(0.4)


# In[289]:


problem_4(0.9)
problem_4(1.1)
problem_4(1.2)


# In[173]:


import math

def problem_5():
    b = 0.4
    for a in np.linspace(0.1, 0.4, 1000):
    # find where the abs value of eigenvalues = 1
        x_right = (b - 1 + math.sqrt(((1 - b) ** 2) + (4 * a))) / 2
        eigenvalue_minus = (-x_right) - math.sqrt((x_right**2) + b)
        eigenvalue_plus = (-x_right) + math.sqrt((x_right**2) + b)

        if eigenvalue_minus < -1 or eigenvalue_plus < -1:
            print(eigenvalue_minus, eigenvalue_plus)
            return a


# In[174]:


problem_5()


# In[279]:


def problem_6():
    iterations = 1000
    b = 0.4
    x_0 = np.asarray((0.8, 0.8))
    fig, axes = plt.subplots(2, 1, figsize=(10, 16))
    
    for a in np.linspace(0.1, 1.2, 100):
        trajectory = iterate_map(henon(a, b), x_0, iterations) # 1000 iterates at a = ___
        axes[0].scatter([a] * (int(iterations / 2) + 1), trajectory[int(iterations / 2):, 0], s=1)
        axes[1].scatter([a] * (int(iterations / 2) + 1), trajectory[int(iterations / 2):, 1], s=1)
    
    axes[0].set_xlabel('a')
    axes[1].set_xlabel('a')
    
    axes[0].set_ylabel('$x_n$')
    axes[1].set_ylabel('$y_n$')
    
    fig.show()


# In[290]:


problem_6()


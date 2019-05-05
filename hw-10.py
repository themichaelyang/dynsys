
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# In[47]:


def problem_3c():
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    
    x = x_grid.flatten()
    y = y_grid.flatten()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(x, y, c=np.arange(len(x)))
    
    g = lambda w, z: (w, z - w**2 / 3)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(g(x, y)[0], g(x, y)[1], c=np.arange(len(x)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    x2 = np.linspace(-5, 5, 100)
    ax.scatter(x2, 1 - np.exp(x2), c=np.arange(len(x2)))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(g(x2, 1 - np.exp(x2))[0], g(x2, 1 - np.exp(x2))[1], c=np.arange(len(x2)))


# In[48]:


problem_3c()


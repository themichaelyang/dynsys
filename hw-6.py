
# coding: utf-8

# In[64]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def iterate_map(f, x_0, n):
    # extended for multi-dim case
#     trajectory = [x_0] * (n+1)
    trajectory = np.stack([x_0] * (n+1), axis=0)
    trajectory[0] = x_0
    for i in range(1, n + 1):
        trajectory[i] = f(*trajectory[i-1])
    
    return trajectory


# In[65]:


henon = lambda a, b: lambda x, y: np.asarray((a - (x**2) + b * y, x))


# In[101]:


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


# In[102]:


problem_4(0.1)
problem_4(0.4)


# In[103]:


problem_4(0.9)
problem_4(1.1)
problem_4(1.2)


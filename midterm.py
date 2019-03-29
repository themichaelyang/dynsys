
# coding: utf-8

# In[18]:


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

def get_plot(f, x_start, x_end, x_step=0.01):
    f_vec = np.vectorize(f)
    x = np.arange(x_start, x_end, x_step)
    y = f_vec(x)
    return np.column_stack((x, y))

identity_fn = lambda x: x


# In[19]:


def problem_1(fn, x_start=0, x_end=1):
    line = get_plot(identity_fn, x_start, x_end)
    plot = get_plot(fn, x_start, x_end)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(plot[:,0], plot[:,1])
    ax.plot(line[:,0], line[:,1])
    
    plt.show()


# In[20]:


problem_1(lambda x: 1 + (2 * x) - (x**2), -2, 3)


# In[30]:


import math
problem_1(lambda x: 0.5 + x + math.sin(x), -5.5, 6.5)


# In[35]:


problem_1(lambda x: x * math.log(abs(x)) if x != 0 else 0, -5, 5)


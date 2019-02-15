
# coding: utf-8

# In[28]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[29]:


def iterate_map(f, x_0, n):
    trajectory = np.zeros(n)
    trajectory[0] = x_0
    for i in range(1, n):
        trajectory[i] = f(trajectory[i-1])
    
    return trajectory

def get_plot(f, x_start, x_end, x_step=0.01):
    f_vec = np.vectorize(f)
    x = np.arange(x_start, x_end, x_step)
    y = f_vec(x)
    return np.column_stack((x, y))

def get_cobweb(trajectory):
    coords = np.column_stack((np.repeat(trajectory[:-1], 2), 
                              np.repeat(trajectory[1:], 2)))
    coords[::2] = np.column_stack((trajectory[:-1], trajectory[:-1]))
    return coords


# In[30]:


logistic_family = lambda a: lambda x: a*x*(1-x)
logistic_map = logistic_family(3.3)

trajectory = iterate_map(logistic_map, 0.1, 10)
cobweb = get_cobweb(trajectory)
plot = get_plot(logistic_map, 0, 1)

identity_fn = lambda x: x
line = get_plot(identity_fn, 0, 1)

print(plot)
print(cobweb)


# In[31]:


fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(cobweb[:,0], cobweb[:,1])
ax.plot(plot[:,0], plot[:,1])
ax.plot(line[:,0], line[:,1])
plt.show()


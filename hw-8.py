
# coding: utf-8

# In[78]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot():
    max_iterations = 100
    z_num = 1000000

    zi = np.zeros(z_num, dtype=complex)
    iterations = np.zeros(z_num, dtype=int)
    
    mandelbrot_map = lambda c: lambda z: z**2 + c
    
    p_num = int(np.sqrt(z_num))
    q_num = z_num // p_num
    
    c = [p + q * 1j for q in np.linspace(-1, 1, q_num)
                    for p in np.linspace(-2, 1, p_num)]
    
    for i in range(max_iterations):
        mask = np.abs(zi) < 2
        
        zi[mask] = (zi ** 2 + c)[mask]
        zi[np.logical_not(mask)] = 2
        iterations[mask] = i + 1

    # p is y axis, q is x axis
    image = iterations.reshape(q_num, p_num)
        
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(image)


# In[81]:


mandelbrot()

# np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
# np.array([y + str(x) for y in ['a', 'b', 'c'] for x in range(4)]).reshape(3, 4)


# In[106]:


def mandelbrot_2():
    max_iterations = 100
    z_num = 15000

    zi = np.zeros(z_num, dtype=complex)
    iterations = np.zeros(z_num, dtype=int)
    
    mandelbrot_map = lambda c: lambda z: z**2 + c
    
    q_p_ratio = 2/3
    p_num = int(np.sqrt(z_num / q_p_ratio))
    q_num = int(q_p_ratio * p_num)
    
    c = [p + q * 1j for q in np.linspace(-1, 1, q_num)
                    for p in np.linspace(-2, 1, p_num)]
    
    for i in range(max_iterations):
        mask = np.abs(zi) < 2
        
        zi[mask] = (zi ** 2 + c)[mask]
        zi[np.logical_not(mask)] = 2
        iterations[mask] = i + 1

    # p is y axis, q is x axis
    image = iterations.reshape(q_num, p_num)
        
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.imshow(image)


# In[107]:


mandelbrot_2()


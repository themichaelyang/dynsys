
# coding: utf-8

# In[21]:


import numpy as np

def mandelbrot():
#     shape = (200, 300)
#     z0 = np.zeros(shape)
    max_iterations = 100
    z_num = 100

    zi = np.zeros(z_num, dtype=complex)
    iterations = np.zeros(z_num, dtype=int)
    
    mandelbrot_map = lambda c: lambda z: z**2 + c
    
    p_num = 10
    q_num = z_num // p_num
    
    # p is y axis
    # q is x axis
    c = [p + q * 1j for q in np.linspace(-1, 1, q_num)
                    for p in np.linspace(-2, 1, p_num)]
    
    for i in range(max_iterations):
        zi[zi < 2] = (zi ** 2 + c)[zi < 2]
        iterations[zi < 2] = i+1
    
    image = iterations.reshape(p_num, q_num)
    
    np.
    
    
mandelbrot()
    


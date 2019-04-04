
# coding: utf-8

# In[9]:


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

henon = lambda a, b: lambda x, y: np.asarray((a - (x**2) + b * y, x))
henon_inv = lambda a, b: lambda x, y: np.asarray((y, (x + y**2 - a) / b))


# In[10]:


# def problem_1():
#     trapezoid = 
def make_polygon(corners, total_pts):
    pts_per_side = total_pts // len(corners)
    points = np.array([]).reshape(0, 2)
    
    for i in range(len(corners)):
        (x_start, y_start) = corners[i]
        (x_end, y_end) = corners[(i+1) % len(corners)]
        
        side_points = np.array([np.linspace(x_start, x_end, pts_per_side), 
         np.linspace(y_start, y_end, pts_per_side)]).T
        
#         np.linspace(x_start, x_end, pts_per_side)
#         np.interp(,  [x_start, x_end], [y_start, y_end])
        points = np.concatenate((points, side_points), axis=0)
    
    return points


# In[ ]:


def recur(fn, x_in, y_in, remaining):
    (x_out, y_out) = fn(x_in, y_in)
    return x if remaining == 0 else recur(fn, x_out, y_out, remaining - 1)

def problem_1():
    a = 4
    b = 0.2
    
    fig1, ax1 = plt.subplots(figsize=(8, 8))
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    
    ax1.set_aspect('equal')
    ax1.set_ylim(-6, 6)
    ax1.set_xlim(-6, 6)
    
    ax2.set_aspect('equal')
    ax2.set_ylim(-6, 6)
    ax2.set_xlim(-6, 6)
    
    trapezoid = make_polygon([[-3, -3], [3, -1.5], [3, 1.5], [-3, 3]], 10**3)
    trapezoid_x = trapezoid[:,0].flatten()
    trapezoid_y = trapezoid[:,1].flatten()
    dist = trapezoid_x + trapezoid_y
    
    # Plot trapezoid, f1 and f1_inv
    ax1.scatter(trapezoid_x, trapezoid_y, c=dist, s=0.1, cmap=plt.get_cmap('cool'))
    
    trapezoid_f1 = henon(a, b)(trapezoid_x, trapezoid_y)
    ax1.scatter(trapezoid_f1[0], trapezoid_f1[1], c=dist, s=0.1, cmap=plt.get_cmap('cool'))
    
    trapezoid_f1_inv = henon_inv(a, b)(trapezoid_x, trapezoid_y)
    ax1.scatter(trapezoid_f1_inv[0], trapezoid_f1_inv[1], c=dist, s=0.1, cmap=plt.get_cmap('cool'))
    
    # Plot trapezoid, f4 and f4_inv
    ax2.scatter(trapezoid_x, trapezoid_y, c=dist, s=0.1, cmap=plt.get_cmap('cool'))
    
    henon4 = lambda a, b: lambda x, y: recur(henon(a, b), x, y, 4)
    henon4_inv = lambda a, b: lambda x, y: recur(henon_inv(a, b), x, y, 4)
    
    trapezoid_f4 = henon4(a, b)(trapezoid_x, trapezoid_y)
    trapezoid_f4_inv = henon4_inv(a, b)(trapezoid_x, trapezoid_y)
    
    ax2.scatter(trapezoid_f4[0], trapezoid_f4[1], c=dist, s=0.1, cmap=plt.get_cmap('cool'))
    ax2.scatter(trapezoid_f4_inv[0], trapezoid_f4_inv[1], c=dist, s=0.1, cmap=plt.get_cmap('cool'))


# In[ ]:


problem_1()


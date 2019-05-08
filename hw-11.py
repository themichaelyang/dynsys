
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt


# In[27]:


def modified_euler(u_start, t_start, t_end, f, t_delta):
    """
    u(t_start) = u_start
    f: u' = f(u)
    """
    u_prev = u_start
    u = [u_prev]
    
    # t values
    steps = np.arange(t_start, t_end + t_delta, t_delta)
    
    for ti in steps:
        u_next_euler = (f(u_prev) * t_delta) + u_prev
        avg_slope = (f(u_next_euler) + f(u_prev)) / 2
        ui = (avg_slope * t_delta) + u_prev
        u_prev = ui
        
        u.append(ui)

    return steps, np.array(u[:-1])


# In[62]:


def part1_problem1():
    starting = [(1, 1, 8),
                (1.01, 1, 8),
                (1.0001, 1, 8),
                (1.000001, 1, 8)]
    
    lorenz = lambda rho, sigma, beta:                 lambda x: np.array([sigma * (x[1] - x[0]), 
                                    rho * x[0] - x[1] - x[0] * x[2],
                                    -beta * x[2] + x[0] * x[1]])
    
    # x is 3 dim
    for pt in starting:
        t, coords = modified_euler(pt, 0, 50, lorenz(28, 10, 8/3), 1/1000)
        x = coords[:,0]
    
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(t, x)


# In[63]:


part1_problem1()


# In[70]:


def part1_problem2():
    starting = [(1, 1, 8),
                (1.01, 1, 8)]
    
    lorenz = lambda rho, sigma, beta:                 lambda x: np.array([sigma * (x[1] - x[0]), 
                                    rho * x[0] - x[1] - x[0] * x[2],
                                    -beta * x[2] + x[0] * x[1]])
    
    # x is 3 dim
    for pt in starting:
        t, coords = modified_euler(pt, 0, 50, lorenz(15, 10, 8/3), 1/1000)
        x = coords[:,0]
    
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(t, x)


# In[71]:


part1_problem2()


# In[85]:


def part1_problem3():
    starting = [(np.sqrt(72), np.sqrt(72), 27),
                (-np.sqrt(72), -np.sqrt(72), 27),
                (0, 0, 0)]
    
    lorenz = lambda rho, sigma, beta:                 lambda x: np.array([sigma * (x[1] - x[0]), 
                                    rho * x[0] - x[1] - x[0] * x[2],
                                    -beta * x[2] + x[0] * x[1]])
    
    # x is 3 dim
    for pt in starting:
        t, coords = modified_euler(pt, 0, 50, lorenz(28, 10, 8/3), 1/1000)
        x = coords[:,0]
    
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(t, x)


# In[86]:


part1_problem3()


# In[87]:


def part1_problem4():
    lorenz = lambda rho, sigma, beta:                 lambda x: np.array([sigma * (x[1] - x[0]), 
                                    rho * x[0] - x[1] - x[0] * x[2],
                                    -beta * x[2] + x[0] * x[1]])
    starting = (1, 1, 8)
    t, coords = modified_euler(starting, 0, 4000, lorenz(28, 10, 8/3), 1/1000)

    x = coords[:,0]
    z = coords[:,2]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, z)


# In[88]:


part1_problem4()


# In[136]:


def part1_problem5():
    lorenz = lambda rho, sigma, beta:                 lambda x: np.array([sigma * (x[1] - x[0]), 
                                    rho * x[0] - x[1] - x[0] * x[2],
                                    -beta * x[2] + x[0] * x[1]])
    starting = (1, 1, 8)
    t, coords = modified_euler(starting, 0, 4000, lorenz(28, 10, 8/3), 1/1000)
    
    z_prev = coords[0][2]
    
    xs = []
    ys = []
    above = []
    
    for pt in coords[1:]:
        x, y, z = pt
        if z < 28 and z_prev > 28:
            ys.append(y)
            xs.append(x)
            above.append(int(y > x))
        z_prev = z
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(xs, ys, c=above[1:] + [1], s=1)


# In[137]:


part1_problem5()


# In[163]:


def rossler(a=0.2, b=0.2, c=5.7):
    def system(u): 
        x, y, z = u
        return np.array([-y - z,
                         x + a*y,
                         b + z*(x-c)])
    return system


# In[186]:


from mpl_toolkits.mplot3d import Axes3D

def plot3d(*args, **kwargs):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(*args, **kwargs)

def part2_problem1():
    system = rossler(b=1.55)
    start = (1, 1, 0)
    
    ts, trajectory = modified_euler(start, 0, 1500, system, 1/100)
    
    # get first index > 1000
    index_1000 = np.argmax(ts > 1000)

    # coords to lists, get trajectory for t > 1000
    xs, ys, zs = zip(*trajectory[index_1000:])
    
    plot3d(xs, ys, zs)


# In[187]:


part2_problem1()


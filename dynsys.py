
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')


import numpy as np
import matplotlib.pyplot as plt


# In[4]:


def f(x):
    return 2 * x

def plot_function():    
    x = np.linspace(-1, 1, 100)
    y = f(x)
    
    plt.plot(x, y)


# In[5]:


plot_function()


# In[8]:


def logistic_map(a, x):
    return a * x * (1-x)

def plot_map():
    x = np.linspace(0, 1, 100)
    y = logistic_map(4, x)
    
    plt.plot(x, y)

plot_map()


# In[12]:


def plot_map2():
    x = np.linspace(0, 1, 100)
    y1 = logistic_map(4, x)
    y2 = logistic_map(4, y1)
    y3 = logistic_map(4, y2)
    
    plt.plot(x, y3)

plot_map2()


# In[47]:


def plot_cobweb():
    x0 = 0.4
    
    cobweb_x = []
    cobweb_y = []
    
    xi = x0
    
    for i in range(100):
        cobweb_x.append(xi)
        cobweb_y.append(xi)

        cobweb_x.append(xi)
        yi = logistic_map(4, xi)
        cobweb_y.append(yi)
        
        xi = yi
        
    plt.plot(cobweb_x, cobweb_y)


# In[48]:


plot_cobweb()


# In[38]:


def theta_circle_x(theta, radius):
    return radius * np.cos(theta)

def theta_circle_y(theta, radius):
    return radius * np.sin(theta)

def plot_circle():
    theta = np.linspace(0, 2 * np.pi)
    
    x = theta_circle_x(theta, 1)
    y = theta_circle_y(theta, 1)
    
    plt.plot(x, y, 'o')


# In[39]:


plot_circle()


# In[44]:


def plot_shape():
    pts_per_side = 100
    
    x1 = 0.5
    y1 = 0.5
    
    x2 = -0.5
    y2 = 0.2
    
    x3 = -0.5
    y3 = -0.2
    
    x4 = 0.5
    y4 = -0.5
    
    side_1_2_x = np.linspace(x1, x2, pts_per_side)
    side_1_2_y = np.linspace(y1, y2, pts_per_side)

    side_2_3_x = np.linspace(x2, x3, pts_per_side)
    side_2_3_y = np.linspace(y2, y3, pts_per_side)
    
    side_3_4_x = np.linspace(x3, x4, pts_per_side)
    side_3_4_y = np.linspace(y3, y4, pts_per_side)
    
    side_4_1_x = np.linspace(x4, x1, pts_per_side)
    side_4_1_y = np.linspace(y4, y1, pts_per_side)
    
    x = [side_1_2_x, side_2_3_x, side_3_4_x, side_4_1_x]
    y = [side_1_2_y, side_2_3_y, side_3_4_y, side_4_1_y]

    plt.plot(x, y, 'o', markersize=1)


# In[45]:


plot_shape()


# In[49]:


def henon_x(a, b, x, y):
    return a - (x**2) + b * y

def henon_y(a, b, x, y):
    return x


# In[61]:


def plot_henon():
    a = 1.4
    b = 0.3
    
    theta = np.linspace(0, 2 * np.pi, 100)
    
    circle_x = theta_circle_x(theta, 1)
    circle_y = theta_circle_y(theta, 1)
    
    x = henon_x(a, b, circle_x, circle_y)
    y = henon_y(a, b, circle_x, circle_y)
    
    plt.plot(x, y, 'o', markersize=1)


# In[62]:


plot_henon()


# In[72]:


def numpy_demo():
    a = np.arange(0, 10, 1)
    
    print(a)
    print(4 * a)
    
    print(np.sin(a))


# In[73]:


numpy_demo()


# In[139]:


def plot_henon4():
    a = 1.4
    b = 0.3
    
    theta = np.linspace(0, 2 * np.pi, 10000)
    
    x0 = theta_circle_x(theta, 1)
    y0 = theta_circle_y(theta, 1)
     
    x = x0
    y = y0
    
    for i in range(4):
        xi = henon_x(a, b, x, y)
        yi = henon_y(a, b, x, y)
        
        x = xi
        y = yi
        
    plt.rcParams['figure.figsize'] = (16, 8)
    plt.plot(xi, yi, 'o', markersize=1)


# In[140]:


plot_henon4()


# In[135]:


def plot_square():
    def f1_x(x, y):
        return x / 2

    def f1_y(x, y):
        return y / 2
    
    pts_per_side = 100
    
    x1 = 0
    y1 = 0
    
    x2 = 1
    y2 = 0
    
    x3 = 1
    y3 = 1
    
    x4 = 0
    y4 = 1
    
    side_1_2_x = np.linspace(x1, x2, pts_per_side)
    side_1_2_y = np.linspace(y1, y2, pts_per_side)

    side_2_3_x = np.linspace(x2, x3, pts_per_side)
    side_2_3_y = np.linspace(y2, y3, pts_per_side)
    
    side_3_4_x = np.linspace(x3, x4, pts_per_side)
    side_3_4_y = np.linspace(y3, y4, pts_per_side)
    
    side_4_1_x = np.linspace(x4, x1, pts_per_side)
    side_4_1_y = np.linspace(y4, y1, pts_per_side)
    
    x = np.array([side_1_2_x, side_2_3_x, side_3_4_x, side_4_1_x]).flatten()
    y = np.array([side_1_2_y, side_2_3_y, side_3_4_y, side_4_1_y]).flatten()

    plt.rcParams['figure.figsize'] = (8, 8)
    plt.plot(x, y, 'o', markersize=1)
    plt.plot(f1_x(x, y), f1_y(x, y), 'o', markersize=1)


# In[136]:


plot_square()


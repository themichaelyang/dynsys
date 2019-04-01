
# coding: utf-8

# In[11]:


# Heads up, future Mike! Turns out this is incorrect for constructing
# the itinerary bounds. Take a look at pg 28 in the textbook. 

# Tent map does have the nice property of halving every time though.
from fractions import Fraction 

itinerary = 'LLRRLRLLR'
bounds = [Fraction(0, 1), Fraction(1, 1)]

for s in itinerary:
    half = (bounds[0] + bounds[1]) / 2
    if s == 'L':
        bounds[1] = half
    else:
        bounds[0] = half

print(bounds)
print(float(bounds[0]), float(bounds[1]))


# In[9]:


itinerary = 'LLRL'
bounds = [0, 1]

for s in itinerary * 1000:
    half = (bounds[0] + bounds[1]) / 2
    if s == 'L':
        bounds[1] = half
    else:
        bounds[0] = half

print(bounds)


# In[14]:


import math 

print(math.sin(math.pi/15) ** 2)


# In[24]:


tent_map = lambda x: 2*x if x <= 0.5 else 2-(2*x)


# In[25]:


def iterate_map(f, x_0, n):
    # careful with n = 1; will iterate 0 times
    # maybe it should be range(1, n + 1)?
    trajectory = [0] * n
    trajectory[0] = x_0
    for i in range(1, n):
        trajectory[i] = f(trajectory[i-1])
    
    return trajectory


iterate_map(tent_map, Fraction(2, 15), 5)


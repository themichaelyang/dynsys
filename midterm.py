
# coding: utf-8

# In[58]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')

import math

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

def get_cobweb(trajectory):
    coords = np.column_stack((np.repeat(trajectory[:-1], 2), 
                              np.repeat(trajectory[1:], 2)))
    coords[::2] = np.column_stack((trajectory[:-1], trajectory[:-1]))
    return coords

identity_fn = lambda x: x


# In[227]:


def problem_1(fn, x_start=0, x_end=1, x_step=0.01):
    line = get_plot(identity_fn, x_start, x_end, x_step)
    plot = get_plot(fn, x_start, x_end, x_step)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    
    ax.plot(plot[:,0], plot[:,1])
    ax.plot(line[:,0], line[:,1])
    
    plt.show()


# In[228]:


problem_1(lambda x: 1 + (2 * x) - (x**2), -2, 3)
print("Fixed points:")
print("{}, source".format(-1.236 / 2))
print("{}, source".format(3.236 / 2))
print("\nDerivatives of fixed points:")
print(abs((lambda x: 2 - 2*x)((1 - math.sqrt(5)) / 2)))
print(abs((lambda x: 2 - 2*x)((1 + math.sqrt(5)) / 2)))


# **Fixed points:**
# 
# $-0.618$, source  
# $1.618$, source

# In[229]:


problem_1(lambda x: 0.5 + x + math.sin(x), -8.5, 8.5)
print(-math.pi / 6)
print(7 * math.pi / 6)
print()
print(1 + (math.sqrt(3)/2)) # source
print(1 - (math.sqrt(3)/2)) # sink


# **Fixed points:**
# 
# $x = {-\pi \over 6} + 2 \pi n,\ n \in \mathbb{Z}$, are sources
# 
# $x = {7\pi \over 6} + 2 \pi n,\ n \in \mathbb{Z}$, are sinks
# 
# Basin of attraction of the sink ${7\pi \over 6} + 2 \pi n$ is:
# 
# $\left({-\pi \over 6} + 2 \pi n, {11 \pi \over 6} + 2 \pi n\right)$
# 
# 
# 

# In[236]:


problem_1(lambda x: x * math.log(abs(x)) if x != 0 else 0, -4, 4)


# **Fixed points:**
# 
# $x = e, -e, 0$ are sources

# In[274]:


def compute_lyapunov_exponent(fn, deriv, val, num_iterates=10**6):
    trajectory = iterate_map(fn, val, num_iterates)
    return np.sum(np.log(np.abs(deriv(trajectory)))) / trajectory.size

# not named find because that implies a search in a set
def compute_lyapunov_number(fn, deriv, val, num_iterates=10**6):
    return math.exp(compute_lyapunov_exponent(fn, deriv, val, num_iterates))


# In[284]:


def problem_2():
    a = 1
    b = 5 / 2
    c = -3 / 2
    
    fn = lambda x: a + b*x + c*(x**2)
    fn_deriv = lambda x: b + 2*c*x
    le_term = lambda x: math.log(abs(fn_deriv(x)))
    
    lyapunov_exp = (1/3) * (le_term(0) + le_term(1) + le_term(2))
    print("a = {}\nb = {}\nc = {}".format(a, b, c))
    print()
    print("Lyapunov number:")
    print(math.exp(lyapunov_exp))
    print()
    print("Lyapunov number (computed iteratively to check):")
    print(compute_lyapunov_number(fn, fn_deriv, [0], 10**6))


# In[285]:


problem_2()
# 1.6355319107919908


# The cycle $(0, 1, 2)$ is a periodic source. The Lyapunov number of 0, which can be thought of as the scaling factor of the trajectory of the neighborhood, is greater than 1, so points in the neighborhood diverge as the map is iteratively applied.

# In[301]:


def problem_3(a=1, x_start=-6, x_end=6, x_step=0.01):
    fn = lambda x: 1 + x + a*math.sin(x)
    
    line = get_plot(identity_fn, x_start, x_end, x_step)
    plot = get_plot(fn, x_start, x_end, x_step)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.grid(True)
    
    ax.plot(plot[:,0], plot[:,1])
    ax.plot(line[:,0], line[:,1])


# In[305]:


problem_3(0.75)
problem_3(1)
problem_3(1.25)


# This is an example of a saddle node bifurication, as fixed points are being created.

# In[447]:


def is_periodic(end, threshold=10**(-6)):
    not_lower_period = all([not math.isclose(end[0], item, rel_tol=threshold)
                            for item in end[1:-1]])
    return not_lower_period and math.isclose(end[0], end[-1], rel_tol=threshold)
        
def problem_4():
    logistic_family = lambda a: lambda x: a*x*(1-x)
    iterates = 1000
    k = 5
    
    for a in np.linspace(3.738, 3.739, 10**5):
        logistic_map = logistic_family(a)
        trajectory = iterate_map(logistic_map, [0.5], iterates)
        
        # get last few iterates
        end = trajectory[:,0][-(k + 1):] 
        if is_periodic(end):
            print(trajectory[:,0][-10:])
            return a
    else:
        return False


# In[448]:


assert is_periodic([0.1, 0.1, 0.1]) == False
assert is_periodic([0.1, 0, 0.1]) == True
assert is_periodic([0.1, 0, 0, 0.100000000000001]) == True


# In[449]:


problem_4()
# 3.738160816081608
# 3.7381732817328173
# 3.7381442814428145
# 3.7381768176817682
# 3.7381768176817682
# 3.738168941689417
# 3.738168561685617
# 3.7381688416884167


# $a = 3.738169$

# In[468]:


from fractions import Fraction 

def problem_6():
    # get bounds of itinerary of tent map
    itinerary = 'LRRRLRL'
    bounds = [Fraction(0, 1), Fraction(1, 1)]

    for s in itinerary:
        half = (bounds[0] + bounds[1]) / 2
        if s == 'L':
            bounds[1] = half
        else:
            bounds[0] = half

    print("Tent map, interval of itinerary: " + itinerary)
    print(bounds)
    print(float(bounds[0]), float(bounds[1]))
    
    # then use conjugacy with logistic map to convert
    print("\nLogistic map (a=4), interval of itinerary: " + itinerary)
    conjugacy_fn = lambda x: math.sin(math.pi * x * 0.5) ** 2
    print(conjugacy_fn(bounds[0]), conjugacy_fn(bounds[1]))


# In[469]:


problem_6()


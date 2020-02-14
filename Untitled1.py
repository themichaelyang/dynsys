
# coding: utf-8

# In[1]:


import numpy as np

hours = np.array([4, 9, 10, 14, 4, 7, 12, 22, 1, 17])
grades = np.array([31, 58, 65, 73, 37, 44, 60, 91, 21, 84])


# In[2]:


x = hours
y = grades


# In[3]:


slope = 3.47


# In[4]:


intersect = 21.693


# In[5]:


n = len(x)


# In[6]:


s_sq =(1 / (n - 2)) * sum((y - intersect - slope * x)**2)


# In[7]:


s_sq


# In[10]:


s_xx = sum((x - np.average(x)) ** 2)


# In[11]:


s_sq / s_xx


# In[16]:


s_sq_biased = (1 / n) * sum((y - intersect - slope * x)**2)
s_sq_biased


# In[13]:


s_sq_biased / s_xx


# In[14]:


np.sqrt(0.05932949202127661)


# In[15]:


np.sqrt(0.0664)


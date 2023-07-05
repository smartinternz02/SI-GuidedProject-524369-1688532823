#!/usr/bin/env python
# coding: utf-8

# In[9]:


name="charan"
age=21
print(name)
print(age)


# In[10]:


X = "Datascience is used to extract meaningful insights."
splitS = X.split()

print(splitS)


# In[13]:


def multiplyNum(a, b):
    return a * b

result = multiplyNum(3,4)
print("Multiplication of num:", result)


# In[14]:


states_capitals = {
    'California': 'Sacramento',
    'Texas': 'Austin',
    'Florida': 'Tallahassee',
    'New York': 'Albany',
    'Ohio': 'Columbus'
}

print("States and Capitals:")
for state, capital in states_capitals.items():
    print(state, "->", capital)


# In[15]:


number_list = list(range(1, 1001))
print(number_list)


# In[16]:


identity_matrix = [[1 if i == j else 0 for j in range(4)] for i in range(4)]
print(identity_matrix)


# In[17]:


matrix = [[j for j in range(i * 3 + 1, (i + 1) * 3 + 1)] for i in range(3)]
print(matrix)


# In[18]:


import numpy as np

array1 = np.array([[1, 2, 3], [4, 5, 6]])
array2 = np.array([[7, 8, 9], [10, 11, 12]])

result = array1 + array2
print("Sum of Arrays:")
print(result)


# In[19]:


import pandas as pd

dates = pd.date_range(start='2023-02-01', end='2023-03-01')
print(dates)


# In[20]:


import pandas as pd

dictionary = {'Brand': ['Maruti', 'Renault', 'Hyundai'], 'Sales': [250, 200, 240]}
df = pd.DataFrame(dictionary)
print(df)


# In[ ]:





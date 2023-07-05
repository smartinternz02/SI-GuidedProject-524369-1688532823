#!/usr/bin/env python
# coding: utf-8

# # FOOD DEMAND PREDICTION

# ## Data Pre-processing
# Data Pre-processing includes the following main tasks
# 
# Import the Libraries.
# Reading the dataset.
# Exploratory Data Analysis
# Checking for Null Values.
# Reading and merging .csv files
# Dropping the columns
# Label Encoding
# Data Visualization.
# Splitting the Dataset into Dependent and Independent variable.
# Splitting Data into Train and Test.

# ### Importing The Libraries

# In[1]:


from pathlib import Path as pth
import pandas as pd
import numpy as np


# ### Reading The Dataset

# In[2]:


train = pd.read_csv("C:\\Users\\VISWA TEJA\\Downloads\\train.csv")
test = pd.read_csv("C:\\Users\\VISWA TEJA\\Downloads\\test.csv")
meal_info = pd.read_csv("C:\\Users\\VISWA TEJA\\Downloads\\meal_info.csv")
center_info = pd.read_csv("C:\\Users\\VISWA TEJA\\Downloads\\fulfilment_center_info.csv")


# ### Exploratory Data Analysis

# In[3]:


train.head()
     


# In[4]:


meal_info.head()


# In[5]:


center_info.head()


# In[6]:


train.info()


# In[7]:


test.info()


# In[8]:


meal_info.info()


# In[9]:


center_info.info()


# In[10]:


train.describe()


# ### Checking For Null Values

# In[11]:


train.isnull().sum()


# ### Reading And Merging .Csv Files

# In[12]:


trainfinal = pd.merge(train, meal_info, on="meal_id", how="outer")


# In[13]:


trainfinal = pd.merge(trainfinal, center_info, on="center_id", how="outer")
trainfinal.head()


# In[14]:


trainfinal.info()


# ### Dropping Columns

# In[15]:


trainfinal = trainfinal.drop(['center_id', 'meal_id'], axis=1)
trainfinal.head()


# In[16]:


cols = trainfinal.columns.tolist()
print(cols)


# In[17]:


cols = cols[:2] + cols[9:] + cols[7:9] + cols[2:7]
print(cols)


# In[18]:


trainfinal = trainfinal[cols]
trainfinal.head()


# In[19]:


trainfinal.dtypes


# ### Label Encoding

# In[20]:


from sklearn.preprocessing import LabelEncoder

lb1 = LabelEncoder()
trainfinal['center_type'] = lb1.fit_transform(trainfinal['center_type'])

trainfinal['category'] = lb1.fit_transform(trainfinal['category'])

trainfinal['cuisine'] = lb1.fit_transform(trainfinal['cuisine'])


# In[21]:


trainfinal.head()


# In[22]:


trainfinal.shape


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns


# ### Data Visualization

# In[24]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 7))
sns.displot(trainfinal.num_orders, bins = 25)
plt.xlabel("num_orders")
plt.ylabel("Number of Buyers")
plt.title("num_orders Distribution")


# In[25]:


trainfinal2 = trainfinal.drop(['id'], axis=1)
correlation = trainfinal2.corr(method='pearson')
columns = correlation.nlargest(8, 'num_orders').index
columns


# In[26]:


correlation_map = np.corrcoef(trainfinal2[columns].values.T)
sns.set(font_scale = 1.0)
heatmap = sns.heatmap(correlation_map, cbar=True, annot=True, square=True, 
                      fmt='.2f', yticklabels=columns.values, 
                      xticklabels=columns.values)
plt.show()


# ### Splitting The Dataset Into Dependent And Independent Variable

# In[27]:


features = columns.drop(['num_orders'])
trainfinal3 = trainfinal[features]
X = trainfinal3.values
y = trainfinal['num_orders'].values


# In[28]:


trainfinal3.head()


# ### Split The Dataset Into Train Set And Test Set

# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25)


# # Modeal building

# ### Train And Test Model Algorithms

# In[30]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor


# In[31]:


from xgboost import XGBRegressor


# ### Model Evaluation

# In[32]:


XG = XGBRegressor()
XG.fit(X_train, y_train)
y_pred = XG.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# In[33]:


LR = LinearRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# In[34]:


L = Lasso()
L.fit(X_train, y_train)
y_pred = L.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# In[35]:


EN = ElasticNet()
EN.fit(X_train, y_train)
y_pred = EN.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# In[36]:


DT = DecisionTreeRegressor()
DT.fit(X_train, y_train)
y_pred = DT.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# In[37]:


KNN = KNeighborsRegressor()
KNN.fit(X_train, y_train)
y_pred = KNN.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# In[38]:


GB = GradientBoostingRegressor()
GB.fit(X_train, y_train)
y_pred = GB.predict(X_val)
y_pred[y_pred<0] = 0
from sklearn import metrics
print('RMSLE:', 100*np.sqrt(metrics.mean_squared_log_error(y_val, y_pred)))


# ### Save The Model

# In[39]:


import pickle
pickle.dump(DT, open('fdemand.pkl','wb'))


# ### Predicting The Output Using The Model

# In[40]:


testfinal = pd.merge(test, meal_info, on="meal_id", how="outer")
testfinal = pd.merge(testfinal, center_info, on="center_id", how="outer")
testfinal = testfinal.drop(['meal_id', 'center_id'], axis=1)

tcols = testfinal.columns.tolist()
tcols = tcols[:2] + tcols[8:] + tcols[6:8] + tcols[2:6]
testfinal = testfinal[tcols]

lb1 = LabelEncoder()
testfinal['center_type'] = lb1.fit_transform(testfinal['center_type'])
testfinal['category'] = lb1.fit_transform(testfinal['category'])
testfinal['cuisine'] = lb1.fit_transform(testfinal['cuisine'])

X_test = testfinal[features].values


# In[41]:


testfinal.dtypes


# In[42]:


pred = DT.predict(X_test)
pred[pred<0] = 0
submit = pd.DataFrame({
    'id' : testfinal['id'],
    'num_orders' : pred
})


# In[43]:


submit.to_csv("submission.csv", index=False)


# In[44]:


submit.describe()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





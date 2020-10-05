#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


dataset = pd.read_csv('Salary_Data.csv')


# In[4]:


dataset.head()


# In[5]:


X= dataset.iloc[:,:-1].values
Y= dataset.iloc[:,1].values


# In[6]:


print(X,Y)


# In[7]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 1/3, random_state = 0)
print(X_train, X_test, Y_train, Y_test)


# In[8]:


from sklearn.linear_model import LinearRegression


# In[9]:


regressor = LinearRegression()
regressor.fit(X_train, Y_train)
print(X_train, Y_train)


# In[10]:


Y_pred = regressor.predict(X_test)


# In[14]:


plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')


# In[17]:


import caTools


# In[ ]:





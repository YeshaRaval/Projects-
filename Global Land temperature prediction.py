#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[2]:


data = pd.read_csv(r"/Users/jayraval/Downloads/GlobalLandTemperatures.csv")


# In[3]:


data.head(5)


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data.AverageTemperature


# In[7]:


data.AverageTemperature.mean


# In[8]:


data.AverageTemperature.fillna(data.AverageTemperature.mean(), inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


data.AverageTemperatureUncertainty


# In[11]:


data.AverageTemperatureUncertainty.mean


# In[12]:


data.AverageTemperatureUncertainty.fillna(data.AverageTemperatureUncertainty.mean(),inplace=True)
                                         
                                                    


# In[13]:


data.isnull().sum()


# In[14]:


features = ['AverageTemperatureUncertainty']
X = data[features]
y = data['AverageTemperature']


# In[15]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2, random_state=30)


# In[16]:


pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('regressor',LinearRegression())
])


# In[17]:


pipeline.fit(X_train,y_train)


# In[18]:


y_pred_linear = pipeline.predict(X_test)


# In[19]:


score = pipeline.score(X_test,y_test)
print(f'R-aquared score is {score} ')


# In[21]:


pred_data = pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred_linear,'Difference':y_test - y_pred_linear})


# In[22]:


pred_data


# In[ ]:





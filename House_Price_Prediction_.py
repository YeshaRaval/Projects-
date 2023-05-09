#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 

data = pd.read_csv(r"https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv")


# In[2]:


data.head(5)


# In[3]:


data.info()


# In[4]:


data.isnull().sum()


# In[5]:


data.total_bedrooms


# In[6]:


df=data
df.total_bedrooms.fillna(df.total_bedrooms.mean() , inplace=True)


# In[7]:


data.isnull().sum()


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[9]:


df.ocean_proximity = df.ocean_proximity.map({'NEAR BAY':1, 'OCEAN':2, 'INLAND':3})


# In[10]:


data.ocean_proximity


# In[11]:


data.isnull().sum()


# In[12]:


df.ocean_proximity.fillna(df.ocean_proximity.median() , inplace=True)


# In[13]:


data.isnull().sum()


# In[14]:


features = ['longitude','latitude','housing_median_age','total_rooms','total_bedrooms','population','households','median_income']
X = data[features]
y = data.median_house_value


# In[15]:


X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.6 , random_state= 16)


# In[16]:


pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('linearegression ', LinearRegression())
])


# In[18]:


pipeline.fit(X_train,y_train)


# In[19]:


y_pred = pipeline.predict(X_valid)


# In[20]:


score = pipeline.score(X_valid,y_valid)
print(f" R-Squared is {score}")


# In[22]:


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[23]:


y_pred


# In[25]:


plt.scatter(y_valid,y_pred);
plt.xlabel('Actual');
plt.ylabel('Predicted');


# In[26]:


pred_data = pd.DataFrame({'Actual Value':y_valid,'Predicted Value':y_pred,'Difference':y_valid - y_pred})


# In[27]:


pred_data


# In[ ]:





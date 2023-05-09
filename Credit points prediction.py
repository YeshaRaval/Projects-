#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 
import numpy as np
df = pd.read_csv(r"/Users/jayraval/Downloads/LOAN.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


pre_df = pd.get_dummies(df,columns=['purpose'],drop_first=True)
pre_df.head()


# In[6]:


from sklearn.model_selection import train_test_split

X = pre_df.drop('not.fully.paid', axis=1)
y = pre_df['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=125
)


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[8]:


pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('logistic_regression ', LogisticRegression())
])


# In[9]:


pipeline.fit(X_train,y_train)


# In[10]:


y_pred = pipeline.predict(X_test)


# In[11]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print(f'accuracy = {accuracy}')


# In[13]:


pred_data = pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test - y_pred})
pred_data


# In[ ]:





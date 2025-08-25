#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import OOPs_deep_stats as deep 
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


df=pd.read_csv("USA_Housing.csv")
df


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


for col_name in df.columns:
    print(col_name)
    st_obj=deep.DataStats_Deep(df[col_name])
    st_obj.everything()


# In[ ]:


df.isnull().sum()


# In[ ]:


X=df.drop('Price', axis=1)
y=df['Price']


# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=555)


# In[ ]:


X_train


# In[ ]:


df.shape


# In[ ]:


print('X train Shape:',x_train.shape)
print('X train Shape:',x_test.shape)
print('y train Shape:',y_train.shape)
print('y train Shape:',y_test.shape)


# In[ ]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)


# In[ ]:


y_pred=lr.predict(x_test)


# In[ ]:


x_test.iloc[0]


# In[ ]:


col_index=0
for i in lr.coef_:
    print(df.columns[col_index])
    print('slope:',round(i,2))
    col_index+=1
    print('*'*30)
print('Y intercept:',lr.intercept_)


# In[ ]:


sse=sum(y_test-y_pred)**2
print('Sum of squared Error:',sse)

mse=sse/len(x_test)
print('Mean squar Error:',mse)

rmse=np.sqrt(mse)
print('Root mean square Error:',rmse)


# In[ ]:


from sklearn


# In[ ]:


sns.displot(y_test-y_pred)


# In[ ]:


for i in df.iloc[2452].values:
    print(i)


# In[ ]:


income=float(input('What is your Income'))
age=float(input('What is your House Age'))
rooms=float(input('What is your Number of Rooms'))
bedroom=float(input('What is your Number of Bedrooms'))
pop=float(input('What is your Area Populstion'))
user_input=np.array([income,age,rooms,bedroom,pop])


# In[98]:


house_price_pred=lr.predict(user_input.reshape(1,-1))
print(f'Your house can be sold between ${round(house_price_pred[0])-5000} to ${round(house_price_pred[0])+5000}')


# In[100]:


pip install joblib


# In[106]:


import joblib
joblib.dump(lr,'usa house price pred model 100k rmse 25-08-25.pkl')


# In[ ]:





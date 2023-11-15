#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[10]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")


# In[12]:


data = pd.read_csv("sales predection.csv",encoding='ISO-8859-1')


# In[13]:


data.head()


# In[14]:


data.info()


# In[17]:


sb.pairplot(data)


# In[18]:


data.isna().sum()


# In[19]:


car_df = data.drop(["customer name","customer e-mail","country"],axis=1)


# In[20]:


Y = car_df[["car purchase amount"]]
X = car_df.drop(["car purchase amount"],axis=1)


# In[21]:


X.shape


# In[22]:


Y.shape


# In[23]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_scaled = mms.fit_transform(X)
Y_scaled = mms.fit_transform(Y.values.reshape(-1,1))


# In[24]:


X_scaled.shape


# In[25]:


Y_scaled.shape


# In[26]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X_scaled,Y_scaled,test_size=0.25,random_state=101)


# In[27]:


xtrain.shape


# In[28]:


ytrain.shape


# In[29]:


xtest.shape


# In[30]:


ytest.shape


# In[44]:


def norm(x):
    return (x - X_stats['mean']) / X_stats['std']

def norm_1(y):
    return (y - y_stats['mean']) / y_stats['std']


# In[45]:


df = pd.read_csv('sales predection.csv',encoding='latin-1')


# In[46]:


df.head()


# In[47]:


df.tail()


# In[48]:


df.describe()


# In[49]:


df.hist()


# In[52]:


# check if there is any null value available or not 
df.isnull().sum()


# In[51]:


New_df = df.drop(['customer name', 'country', 'customer e-mail'], axis=1)
New_df.shape


# In[53]:


# define x and y using iloc function 
x = New_df.iloc[:,:-1]
y = New_df.iloc[:,-1]



x.shape 


# In[54]:


# to normalization we need mean value and standard deviation 
# here , i am not doing normalization by sklearn library 

X_stats = x.describe().transpose()
X_stats

y_stats = y.describe().transpose()


# In[61]:


X = norm(x)
Y = norm_1(y)
X = np.array(X)
Y = np.array(Y)


# In[60]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.25,random_state=42)


# In[62]:


def plot_diff(y_true, y_pred, title=''):
    plt.figure(figsize=(4,4),dpi=150)
    plt.scatter(y_true, y_pred,color='blue')
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-10, 10], [-10, 10],color='red')
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.figure(figsize=(4,4),dpi=150)
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()


# In[63]:


Y_pred = model.predict(X_test)
plot_diff(y_test, Y_pred, title='True Vs predicted values')


# In[ ]:





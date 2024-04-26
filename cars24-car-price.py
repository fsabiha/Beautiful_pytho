#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[428]:


## Importing libraries


# In[429]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:





# In[430]:


## Downloading the dataset


# In[431]:


df=pd.read_csv('cars24-car-price-cleaned.csv')
df.head()


# In[ ]:





# In[432]:


df.shape


# In[433]:


df['model'].nunique()


# In[434]:


df['make'].nunique()


# In[435]:


df['make'].value_counts()


# In[ ]:





# ### Prepocessing of data

# In[436]:


## Converting categorical cols to num cols - using target encoding


# In[437]:


df['make'] = df.groupby('make')['selling_price'].transform('mean')
df['model'] = df.groupby('model')['selling_price'].transform('mean')
df.head()


# In[ ]:





# In[438]:


## scaling - standardising & normalisation (here scaling shoudnt be done on selling_price as it is the target col)


# In[439]:


#from sklearn.preprocessing import StandardScaler, MinMaxScaler

#scaler=MinMaxScaler()
#df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns)
#df.head()


# In[440]:


## Assigning the labels(x & y) to the target col & other col


# In[441]:


X=df.drop('selling_price',axis=1)
y=df['selling_price']
X.shape,y.shape


# In[ ]:





# In[442]:


## creating train_test_split - dividing the data into train data and test data


# In[443]:


## Random data points are split 


# In[444]:


from sklearn.model_selection import train_test_split


# In[445]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[446]:


X_train.shape, y_train.shape


# In[447]:


X_test.shape, y_test.shape


# In[ ]:





# ## Train the algorithm with X_train & y_train
# 
# 
# 
# ## check the performance on the test data

# In[448]:


## Univeriate - Lin Reg model


# In[449]:


df.corr() ## checking the correlation of the features on target col


# In[ ]:





# In[450]:


X_train1 = X_train[['model']]  ## selected 'model' col based on the highest corr
X_test1 = X_test[['model']]


# In[451]:


plt.scatter(X_train1,y_train)
plt.show()


# In[ ]:





# In[452]:


## modelling 


# In[453]:


from sklearn.linear_model import LinearRegression


# In[454]:


model = LinearRegression() ## creaing the lin_reg model


# In[455]:


model.fit(X_train1, y_train) ## fitting the lin_reg model(# model is trained # the perfect line is found.)


# In[456]:


## weights

model.coef_ 


# In[457]:


model.intercept_ ## Slope


# In[ ]:





# In[458]:


## y = 0.99*x + 0.003 (slope formula y=mx+c)


# In[ ]:





# In[459]:


## making the model predict on a single value


# In[460]:


model.predict([[2]])


# In[461]:


## making the model predict on train data


# In[462]:


model.predict(X_train1)


# In[463]:


## making the model predict on range of 0 & 1


# In[464]:


model.predict([[0],[1]])


# In[465]:


## Plotting the line for 0,20


# In[466]:


plt.scatter(X_train1,y_train)
plt.plot([0,20],model.predict([[0],[20]]),c='orange')
plt.show()


# In[ ]:





# In[467]:


## plotting it on the X_train1 data


# In[468]:


plt.scatter(X_train1,y_train)
plt.plot(X_train1,model.predict(X_train1),c='orange')
plt.show()


# In[469]:


model.predict(X_train1)


# In[470]:


## checking the values for first 10 values

model.predict(X_train1)[:10]


# In[471]:


y_train[:10]


# In[ ]:





# In[472]:


## Evaluate the model


# In[473]:


model.score(X_train1, y_train)


# In[474]:


model.score(X_test1, y_test)


# In[ ]:





# In[475]:


## MAE & MSE


# In[476]:


# MAE
np.abs( y_train - model.predict(X_train1) ).mean()


# In[477]:


# MSE
np.square( y_train - model.predict(X_train1) ).mean()


# In[ ]:





# ## Multi-variate Linear Regression

# In[478]:


from sklearn.linear_model import LinearRegression


# In[479]:


X_train.head()


# In[480]:


y_train.head()


# In[481]:


## Scaling/standardising the data


# In[ ]:





# In[482]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# scaler = MinMaxScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
# X_train.head()


# In[483]:


scaler = MinMaxScaler()
scaler.fit_transform(X_train)


# In[484]:


scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_train.head()


# In[485]:


X_test = pd.DataFrame (scaler.transform(X_test), columns=X_test.columns)
X_test.head()


# In[486]:


## Training the model after scaling

model=LinearRegression()
model.fit(X_train,y_train)


# In[487]:


model.coef_.round(2)


# In[488]:


model.intercept_


# In[489]:


model.score(X_train, y_train)


# In[490]:


model.score(X_test, y_test)


# In[ ]:





# In[ ]:


# Feature selection


# In[491]:


import seaborn as sns

imps = np.abs(model.coef_).round(3)
imps


# In[492]:


sns.barplot(x = X_train.columns, y = imps)
plt.xticks(rotation=90)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# # Import Necessary Libraries 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# # EDA

# In[2]:


df = pd.read_csv('D:\mlops\sajat\data\spam.csv')
df.head(20)


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df.value_counts()


# In[6]:


df[df['Category'] == 'spam'].value_counts()


# In[7]:


df[df['Category'] == 'ham'].value_counts()


# **after analysing the data, we will need to :**
# 1. replace the 'category' column with numeric values
# 2. use countVectorizer to represent each data as a vector of numbers

# # Data Preprocessing

# In[8]:


# Making a 'spam' column and replacing spam with 1 
df['spam'] = df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


# In[9]:


#droping the category column
df.drop('Category',inplace =True, axis =1)


# ## Train Test Split

# In[10]:


x= df.Message
y= df['spam']


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# # Building The Model

# In[12]:


clf = Pipeline([('vectorizer', CountVectorizer()),('nb', MultinomialNB())])


# In[13]:


clf.fit(x_train,y_train)


# ## The Model Accuracy

# In[14]:


clf.score(x_train,y_train)


# In[15]:


clf.score(x_test,y_test)


# In[16]:


testing_emails = ['hi, wanna hangout at 10? i heard there is a good chinese restaurant nearby. see you soon'
                 ,"don't miss this chance to win 100$ dollars"]


# In[17]:


clf.predict(testing_emails)


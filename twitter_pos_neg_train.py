#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
# tokenizing in various ways
from nltk.tokenize import word_tokenize
# stopwwords collection
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pickle
# lemmatize like stem
from nltk.stem import WordNetLemmatizer
 
# To Wrap up the sklearn classifiers to be used in NLP for classifying
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.metrics import *
from sklearn.naive_bayes import MultinomialNB

import numpy as np
import pandas as pd
import sys
import os


# In[ ]:


a = sys.stdin.read()


# In[2]:


train_data, model_path = a.split()


# In[3]:


# train_data = "Data/Sentiment Analysis Dataset.csv"
# model_path = "model_pickle/classifier.pkl"


# In[4]:


train_df =pd.read_csv(train_data)


# In[5]:


train_df.head()


# In[6]:


train_df=train_df[["Sentiment","SentimentText"]]


# In[7]:


train_df.head(3)


# In[8]:


# Removes stopwords and storing only storing alpabets and numbers and imp symbols also lemamtizing to decrease number of features
class Preprocessor:

    def preprocessor(self,doc):
        lm = WordNetLemmatizer()
        preprop = lambda x: ' '.join([lm.lemmatize(word) for word in x.split() if word not in stopwords.words('english') and not(word.isalpha() or word.startswith('@') or word.isnumeric() or (word in ['!','.',','])) ])
        return doc.apply(preprop)


# In[9]:


pre_process = Preprocessor()


# In[10]:


train_df= train_df.dropna()


# In[11]:


train_df.isnull().sum()


# In[12]:


train_df = train_df.drop_duplicates()


# In[13]:


train_df = train_df.drop(train_df[(train_df["SentimentText"]=="&quot")].index.values, axis=0) 
# Delete all rows with label "Ireland"


# In[14]:


train_df.shape[0]


# In[15]:


# storing preprocessor object


# In[16]:


class by_count_vectorizer:
    
    def check_result(self,feature,classifier):
        if(classifier.predict(feature)==1):
            print('Good')
        else:
            print('Hate')
    
    def accuracy(self,Y_actual,Y_pred):
        correct = (Y_actual==Y_pred).sum()
        return (correct/y_actual.shape[0])*100
    
    def classify_demo(self,train_df):
        pre_process = Preprocessor()
        train_df["SentimentText"] = pre_process.preprocessor(train_df["SentimentText"])
        cv = CountVectorizer()
        X_train,Y_train = train_df["SentimentText"],train_df["Sentiment"]
        X_train,X_cross,Y_train,Y_cross = train_test_split(X_train,Y_train,test_size=0.1)
        X_train = cv.fit_transform(X_train)
#         print("to array:\n",X_train.toarray())
        mnb = MultinomialNB()
        mnb.fit(X_train,Y_train)
        y_pred = mnb.predict(X_cross)
        print("Accuracy: ",self.accuracy(Y_cross,y_pred))
        file = open(model_path,'wb')
        pickle.dump(pre_process,file)
        pickle.dump(cv,file)
        pickle.dump(mnb,file)
        file.close()
        try:
            sys.stdout.write(pred)
        except Exception as e:
            print("Cannot return the path beacause ",e)

#         feature = cv.transform(["The movie was pleasant"])
#         self.check_result(feature,mnb)


# In[17]:


cv_classifier = by_count_vectorizer()


# In[ ]:


cv_classifier.classify_demo(train_df)


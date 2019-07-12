#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[4]:


a = sys.stdin.read()
folder ,model_path , train , test = a.split()


# In[162]:


train_df =pd.read_csv(folder+"/"+train)
test_df = pd.read_csv(folder+"/"+test)


# In[163]:


train_df.head()


# In[173]:


test_df.head(10)


# In[165]:


train_df=train_df[["label","tweet"]]


# In[166]:


train_df.head(3)


# In[167]:


test_df.head(3)


# In[168]:


# Removes stopwords and storing only storing alpabets and numbers and imp symbols also lemamtizing to decrease number of features
class Preprocessor:

    def preprocessor(self,doc):
        lm = WordNetLemmatizer()
        preprop = lambda x: ' '.join([lm.lemmatize(word) for word in x.split() if word not in stopwords.words('english') and not(word.isalpha() or word.startswith('@') or word.isnumeric() or (word in ['!','.',','])) ])
        return doc.apply(preprop)


# In[150]:


pre_process = Preprocessor()


# In[151]:


# train_df["tweet"] = pre_process.preprocessor(train_df["tweet"])


# In[152]:


# test_df['tweet']=pre_process.preprocessor(test_df['tweet'])


# In[153]:


# train_df.head(2)


# In[154]:


# test_df.head(2)


# In[155]:


# print(train_df.shape,test_df.shape)


# In[156]:


# storing preprocessor object


# In[169]:


class by_count_vectorizer:
    
    def check_result(self,feature,classifier):
        if(classifier.predict(feature)==1):
            print('Good')
        else:
            print('Hate')
    
    def classify_demo(self,train_df,test_df):
        pre_process = Preprocessor()
        train_df["tweet"] = pre_process.preprocessor(train_df["tweet"])
        test_df['tweet']=pre_process.preprocessor(test_df['tweet'])
        cv = CountVectorizer()
        X_train,Y_train = train_df["tweet"],train_df["label"]
        X_train = cv.fit_transform(X_train)
        print("to array:\n",X_train.toarray())
        x_cross = test_df["tweet"]   
        mnb = MultinomialNB()
        mnb.fit(X_train,Y_train)
        y_pred = mnb.predict(x_cross)
        print("Accuracy by f1 score ", f1_score(y_cross,y_pred)*100)
        file = open(model_path+'/classifier.pkl','wb')
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


# In[170]:


cv_classifier = by_count_vectorizer()


# In[5]:


cv_classifier.classify_demo(train_df,test_df)


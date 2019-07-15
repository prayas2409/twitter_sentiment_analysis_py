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
train,model_path = a.split()

# In[162]:

train_df =pd.read_csv(train)

# In[163]:

# train_df.head()

train_df=train_df[["label","tweet"]]

# In[166]:

#train_df.head(3)

# Removes stopwords and storing only storing alphabets and numbers and imp symbols also lemamtizing to decrease number of features
class Preprocessor:

    def preprocessor(self,doc):
        lm = WordNetLemmatizer()
        preprop = lambda x: ' '.join([lm.lemmatize(word) for word in x.split() if word not in stopwords.words('english') and not(word.isalpha() or word.startswith('@') or word.isnumeric() or (word in ['!','.',','])) ])
        return doc.apply(preprop)

# In[150]:
pre_process = Preprocessor()

class by_count_vectorizer:
    
    def check_result(self,feature,classifier):
        if(classifier.predict(feature)==1):
            print('Good')
        else:
            print('Hate')
    
    def classify_demo(self,train_df,test_df):
        pre_process = Preprocessor()
        train_df["tweet"] = pre_process.preprocessor(train_df["tweet"])
        cv = CountVectorizer()
        X_train,Y_train = train_df["tweet"],train_df["label"]
        print("to array:\n",X_train.toarray())
        x_cross = test_df["tweet"]   
        mnb = MultinomialNB()
        mnb.fit(X_train,Y_train)
#        print("Accuracy by f1 score ", f1_score(y_cross,y_pred)*100)
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

cv_classifier = by_count_vectorizer()

cv_classifier.classify_demo(train_df,test_df)


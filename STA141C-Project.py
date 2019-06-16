#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import math
import numpy as np
import time

from collections import defaultdict



from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor


import nltk
#import xlrd
import string
import nltk.corpus
from nltk.corpus import wordnet

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import re
nltk.download('stopwords')
import nltk.corpus as corpus
nltk.download('vader_lexicon')
import re

stopwords = corpus.stopwords.words("english")

import ast 
import statistics
from statistics import mean
import itertools
from itertools import chain
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[7]:


def get_sentiment(des):
    
    des = list(des)
    
    def get_adj_and_adv(text):
        """
        This functionis to firstly tokenize the words and then select
        the words that is tagged as adverbe and adjective
        """
        text_lower = text.lower()
        text_token = word_tokenize(text_lower)
        result_tags = nltk.pos_tag(text_token)
    
        words = [(word) for word, tag in result_tags if tag in ('JJ','RB')]
        
        return (words)
    
    def get_noun(text):
        """
        This functino is to tokenize the words and select the words
        that is tagged as noun
        """
        text_lower = text.lower()
        text_token = word_tokenize(text_lower)
        result_tags = nltk.pos_tag(text_token)
    
        words = [(word) for word, tag in result_tags if tag in ('NN')]
        return (words)
    
    def nltk_sentiment(sentence):
        """
        This function is to process the sentiment on each tokenized sentences
        and then generate a sentiment value for each sentence
        """
    
        nltk_sentiment = SentimentIntensityAnalyzer()
        score = nltk_sentiment.polarity_scores(sentence)
        return score
    
    sen_tok = [sent_tokenize(des[i]) for i in range(len(des))]

    sen_tok_total = [''.join(sen_tok[i]) for i in range(len(sen_tok))]

    x = [nltk_sentiment(sen_tok_total[i]) for i in range(len(sen_tok_total))]

    x1 = [(list(x[i].items())[-1][1]) for i in range(len(x)) ]
    
    return x1


# In[8]:


def missing_value(df):
    """
    Handeling any missing value in the dataframe.
    """
    df["item_description"][df['item_description'] == "No description yet"] = "None"
    return df.fillna("None")

def split_label(cat):
    """
    This function splits the category into three sub categories.
    """
    cat_split = cat.str.split("/",n = 2,expand = True)
    cat_split = cat_split.rename(index = str,columns = {0:'cat1',1:'cat2',2:'cat3'})
    cat_split = cat_split.fillna("None")
    
    return cat_split


def variable_process(df):

    sub_cats = split_label(df["category_name"])
    
    columns = (df[["shipping"]].values, df[["brand_name"]].values, df[["item_condition_id"]].values, 
               sub_cats.values)
    columns_names = ("shipping", "brand_name", "item_condition", "cat1", "cat2", "cat3")


    return pd.DataFrame(np.concatenate(columns, axis = 1), columns = columns_names)
    

def encode(sub_cat_train):
    """
    This function one hot encode category variables for the training set and the testing set.
    """
    
    from sklearn.preprocessing import OneHotEncoder
    
    global onehotencoder
    
    onehotencoder = OneHotEncoder(handle_unknown='ignore')
    
    one_hot_train = onehotencoder.fit_transform(sub_cat_train.values).toarray()

    return one_hot_train

def linear_encoder(sub_cat):
    le = preprocessing.LabelEncoder()
    label_cat = le.fit_transform(sub_cat)
    return label_cat
    

def get_length_of_des(des):
    #des = list(df['item_description'])
    text_token = [word_tokenize(x) for x in des]
    length = [len(t) for t in text_token]
    
    return length

def price(df):
    price = df['price'].values
    return np.log(price+1)
    


# In[9]:


# Return: the input for modeling
# !!! data_preparation_train() Only for TRAIN DATA

def data_preparation_train(df):

    df = missing_value(df)
    df_label = split_label(df["category_name"])
    df1 = variable_process(df)
    df2 = encode(df1)

    df["sent"] = get_sentiment(df["item_description"])
    df2 = np.append(df2, df["sent"].values.reshape(-1,1), axis=1)
    
    df["length"] = get_length_of_des(df["item_description"])
    df2 = np.append(df2, df["length"].values.reshape(-1,1), axis=1)

    return df2


# data_preparation_test only for test 

def data_preparation_test(df):

    df = missing_value(df)
    df_label = split_label(df["category_name"])
    df1 = variable_process(df)
    df2 = onehotencoder.transform(df1).toarray()

    df["sent"] = get_sentiment(df["item_description"])
    df2 = np.append(df2, df["sent"].values.reshape(-1,1), axis=1)
    
    df["length"] = get_length_of_des(df["item_description"])
    df2 = np.append(df2, df["length"].values.reshape(-1,1), axis=1)

    return df2



# In[17]:


# # DATA PREP FOR LINEAR REGRESSION ONLY !!!

def data_preparation_linear(df):
    
    df = missing_value(df)
    df1 = variable_process(df)
    
    d = defaultdict(preprocessing.LabelEncoder)
    df2 = df1.apply(lambda x: d[x.name].fit_transform(x))
    
    df["sent"] = get_sentiment(df["item_description"])
    df2 = np.append(df2, df["sent"].values.reshape(-1,1), axis=1)
    
    df["length"] = get_length_of_des(df["item_description"])
    df2 = np.append(df2, df["length"].values.reshape(-1,1), axis=1)

    return df2


# In[21]:


def models(X_train, X_test, X_trainl, X_testl, y_train, y_test):
    
    # LINEAR REGRESSION
    start_reg_fit = time.time()
    reg = LinearRegression().fit(X_trainl, y_train)
    end_reg_fit = time.time()
    time_reg_fit = end_reg_fit-start_reg_fit
    
    # RANDOM FOREST
    start_rf_fit = time.time()
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf.fit(X_train, y_train)
    end_rf_fit = time.time()
    time_rf_fit = end_rf_fit-start_rf_fit

    # LASSO
    start_cl_fit = time.time()
    clf = linear_model.Lasso(alpha=0.1)
    clf.fit(X_train, y_train)
    end_cl_fit = time.time()
    time_cl_fit = end_cl_fit-start_cl_fit

    # KNN
    start_knn_fit = time.time()
    neigh = KNeighborsRegressor(n_neighbors=3)
    neigh.fit(X_train, y_train)
    end_knn_fit = time.time()
    time_knn_fit = end_knn_fit-start_knn_fit
    
    # predictions
    start_reg_pred = time.time()
    reg_pred = reg.predict(X_testl)
    end_reg_pred = time.time()
    time_reg_pred = end_reg_pred - start_reg_pred
    
    start_rf_pred = time.time()
    rf_pred = rf.predict(X_test)
    end_rf_pred = time.time()
    time_rf_pred = end_rf_pred-start_rf_pred
    
    
    start_cl_pred = time.time()
    cl_pred = clf.predict(X_test)
    end_cl_pred = time.time()
    time_cl_pred = end_cl_pred-start_cl_pred 
    
    start_knn_pred = time.time()
    knn_pred = neigh.predict(X_test)
    end_knn_pred = time.time()
    time_knn_pred = end_knn_pred - start_knn_pred
    
    time_reg = time_reg_fit+time_reg_pred
    time_rf = time_rf_fit+time_rf_pred
    time_cl = time_cl_fit+time_cl_pred
    time_knn = time_knn_fit+time_knn_pred

    dic = {
        "reg_pred" : reg_pred,
        "rf_pred" : rf_pred,
        "cl_pred" : cl_pred,
        "knn_pred": knn_pred,
        "y_test" : y_test,
        
        
    }
    
    dic_time = {
        
        "reg_time" : [time_reg],
        "rf_time" : [time_rf],
        "cl_time" : [time_cl],
        "knn_time" : [time_knn]
    }
    
    print(pd.DataFrame(dic_time))
    
    df = pd.DataFrame(dic)
    
    df["mean"] = df[["reg_pred","rf_pred", "cl_pred", "knn_pred"]].mean(axis = 1)
    
    return df


# In[ ]:





# In[22]:


def main():
    
    df = pd.read_csv("/Users/apple/Desktop/untitled folder/train.tsv/train.tsv", delimiter = "\t", encoding = "utf-8", index_col = False).iloc[:100000, :]
    #df = df.iloc[:100000, :]
    train, test, ytrain, ytest = train_test_split(df, df["price"], test_size=0.3, random_state = 0)
    
    X_train = data_preparation_train(train)
    X_test = data_preparation_test(test)
    
    y_train = np.log(ytrain + 1)
    y_test = np.log(ytest + 1)
    
    X_trainl = data_preparation_linear(train)
    X_testl = data_preparation_linear(test)
    
    def mse(preds, actual = y_test):
        return np.sqrt(mean_squared_error(actual, preds))
    
    results = models(X_train, X_test, X_trainl, X_testl, y_train, y_test)
    
    print(results[["reg_pred", "rf_pred", "cl_pred", "knn_pred", "mean"]].apply(lambda x: mse(x)))
    
    return
    
    


# In[23]:

if __name__ == '__main__':
    main()




# In[ ]:





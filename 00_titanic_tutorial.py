# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 15:36:19 2019

@author: MA063543
"""

# tutorial on python using kaggle titanic
# https://www.kaggle.com/startupsci/titanic-data-science-solutions

# basic data
import os 

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# get data
path = os.getcwd()
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

# print column names
col = train_df.columns.values
print(col)

# preview
train_df.head()
train_df.tail()
train_df.shape

print('train data\n')
train_df.info() # like str() in r
print('_'*40,'\n') # create a line break
print('test data\n')
test_df.info()

round(train_df.describe()) # like summary() in r
train_df.describe(include=['O']) # summarize categorical vars

# means by group
train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
round(train_df[['Sex', 'Survived']].groupby('Sex',  as_index = False).mean(),3)
train_df[["Parch", "Survived"]].groupby('Parch').mean().sort_values(by = 'Survived', ascending = False)

# viz
g = sns.FacetGrid(train_df, col = 'Survived')
g.map(plt.hist, 'Age', bins = 20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha = .5, bins = 20)
grid.add_legend()

### wrangle data
train_df = train_df.drop(['Ticket', 'Cabin'], axis = 1)
train_df.columns
test_df = test_df.drop(['Ticket', 'Cabin'], axis = 1)
test_df.columns.values

combine = [train_df, test_df]

# identify titles in names using The RegEx pattern (\w+\.) matches the first 
# word which ends with a dot character within Name feature. The expand=False flag returns a DataFrame.

for dataset in combine:
    ''' add title columnbn to both dfs ''' 
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
    
pd.crosstab(train_df['Title'], train_df['Sex'])

for dataset in combine:
    ''' replace unusual titles with 'Rare' label '''
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', \
           'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean().sort_values(by = 'Survived', ascending = False)

# convert titles to ordinal
title_mapping = {"Mr":1, "Miss":2, "Mrs":3, "Master":4, "Rare":5}
for dataset in combine:
    '''convert to ordinal column'''
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# drop Name column
train_df = train_df.drop(['Name', 'PassengerId'], axis = 1)
test_df = test_df.drop(['Name', 'PassengerId'], axis = 1)
combine = [train_df, test_df]
combine[0].shape, combine[1].shape

# change sex var
for dataset in combine:
    ''' 1 = female and 0 = male'''
    dataset['Sex'] = dataset['Sex'].map( {"female": 1, "male": 0}).astype(int)
    
train_df.head()
# left off at [23] in kaggle notebook
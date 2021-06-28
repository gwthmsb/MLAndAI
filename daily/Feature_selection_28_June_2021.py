# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 22:23:21 2021

@author: gowthas

https://towardsdatascience.com/four-popular-feature-selection-methods-for-efficient-machine-learning-in-python-fdd34762efdb

"""

import pandas as pd
import numpy as np

df = pd.read_csv('D:\docs\GITLab\ML & AI\datasets\daily_exercises\Feature_Selection_28_june_2021.csv')

print(df.columns)

X = df.drop('havarth3', axis=1)
y = df['havarth3']


"""
Univariate Feature Selection

Selects best n features from dataset

Score functions are used to score each feature
1) f_classif
2) chi2
3) mutual_info_classif
"""

from sklearn.feature_selection import SelectKBest, f_classif

uni = SelectKBest(score_func=f_classif, k=10)
fit = uni.fit(X, y)

# Print the best scored indices.
print(fit.get_support(indices=True))

# Printed columns names are best features
print(X.columns[fit.get_support(indices=True)])



"""
Corellation matrix

This process calculates the correlations of all features with target features
A threshold is defined, if correlation exceeds that threshold, then that feature
is selected

"""

cor = df.corr()
cor_target = abs(cor['havarth3'])
relavant_features = cor_target[cor_target > 0.2]    
# havarth3 will also comes under relavant_features. Needs to remove it
r_features = relavant_features.drop('havarth3')


"""
Principal Component Method(PCA)

PCA explains which features explain the maximum variance

"""

from sklearn.decomposition import PCA

model = PCA(n_components=10).fit(X)
n_pcs = model.components_.shape[0]
most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
df.columns[most_important]


"""
Wrapper

In this method, one machine learning method is used to find the right features.
This method uses the p-value.

"""

import statsmodels.api as sm

X_new = sm.add_constant(X)
model = sm.OLS(y, X_new).fit()
print(model.pvalues)

selected_features = list(X.columns)

while(len(selected_features) > 0):
    p = []
    X_new = X[selected_features]
    X_new = sm.add_constant(X_new)
    model = sm.OLS(y,X_new).fit()
    p = pd.Series(model.pvalues.values[1:],index = selected_features)      
    pmax = max(p)
    feature_pmax = p.idxmax()
    if(pmax>0.05):
        selected_features.remove(feature_pmax)
    else:
        break
    
print(selected_features)

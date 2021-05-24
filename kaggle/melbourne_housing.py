# -*- coding: utf-8 -*-
"""
Dataset: Melbourne housing snapshot
Path: D:\docs\GITLab\ML & AI\datasets\melbourne-housing-snapshot
URL: https://www.kaggle.com/dansbecker/melbourne-housing-snapshot

@author: gowthas
"""

"""
Prediction target, y: Column which we want to predict
Choosing features: Features on which prediction is modelled
Data, X: By convention

Model validation
    in-sample score: When same data is used for training and prediction

Overfitting and Underfitting

Random Forest: https://www.kaggle.com/dansbecker/random-forests

"""


import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# showing all columns and rows
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)

# Reading dataset

home_data = pd.read_csv("D:\docs\GITLab\ML & AI\datasets\melbourne-housing-snapshot\melb_data.csv")
print(home_data.describe())
print(home_data.columns)

home_data = home_data.dropna(axis=0)

y = home_data.Price
print(y)

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = home_data[melbourne_features]
print(X)

model = DecisionTreeRegressor(random_state=1)
model.fit(X, y)
predicted_values = model.predict(X)


"""
Model validation

1) Mean Absolute Error(MAE): On average, our predictions are off by about X
With the MAE metric, we take the absolute value of each error. This converts 
each error to a positive number. We then take the average of those absolute errors. 
This is our measure of model quality


"""

mean_abs_error = mean_absolute_error(y, predicted_values)
print(mean_abs_error)

"""
Train and test data split
 
"""

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model = DecisionTreeRegressor(random_state = 1)
model.fit(train_X, train_y)

predicted_values = model.predict(val_X)
mean_abs_error_2 = mean_absolute_error(val_y, predicted_values)

print(mean_abs_error_2)


"""
Overfitting and Underfitting

"""

def mae_model(max_leaves, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(random_state=1, max_leaf_nodes= max_leaves)
    model.fit(train_X, train_y)
    predicted_values = model.predict(val_X)
    mae = mean_absolute_error(val_y, predicted_values)
    return mae

for max_leaves in [5, 50, 500, 5000]:
    mae = mae_model(max_leaves, train_X, val_X, train_y, val_y)
    print("Max leaves: {} \t\t MAE: {}".format(max_leaves, mae))
    

"""
Random Forest
"""
model = RandomForestRegressor(random_state=1, max_leaf_nodes= 500)
model.fit(train_X, train_y)

predicted_values = model.predict(val_X)
mae = mean_absolute_error(val_y, predicted_values)
print(mae)

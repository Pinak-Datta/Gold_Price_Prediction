# Importing dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# Data collection and preprocessing

gold_data = pd.read_csv('/content/gold price dataset.csv')
gold_data.head()
gold_data.isnull().sum()


#Splitting the dataset into target and features

X = gold_data.drop(['Date','GLD'],axis=1)
Y = gold_data['GLD']


#Splitting X and Y into training and testing values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=2)

# Model Definition and training

regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(X_train,Y_train)

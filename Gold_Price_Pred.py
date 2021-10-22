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

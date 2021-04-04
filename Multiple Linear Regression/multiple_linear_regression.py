# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_excel("Company Data.xlsx")
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [3])
X[:,3] = labelencoder_X.fit_transform(X[:,3])
X = onehotencoder.fit_transform(X).toarray()


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size= 0.2,random_state = 65)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred = regressor.predict(X_test)

import statsmodels.api as stm 
X = np.append(arr =  np.ones((50,1)).astype(int),values = X,axis = 1)
X_best = X[:,[0,1,2,3,4,5,6]]
regressor_OLS = stm.OLS(endog = Y, exog = X_best).fit()
regressor_OLS.summary()

X_best = X[:,[0,1,2,3,4,6]]
regressor_OLS = stm.OLS(endog = Y, exog = X_best).fit()
regressor_OLS.summary()

X_best = X[:,[0,1,2,3,4]]
regressor_OLS = stm.OLS(endog = Y, exog = X_best).fit()
regressor_OLS.summary()
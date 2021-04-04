# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset  = pd.read_excel("Decision Tree Regression.xlsx")
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 62)
regressor.fit(X,Y)


y_pred = regressor.predict([[6.5]])


X_hr = np.arange(min(X),max(X),0.01)
X_hr = X_hr.reshape((len(X_hr),1))
plt.scatter(X,Y,color="blue")
plt.plot(X_hr,regressor.predict(X_hr),color="red")
plt.title("Finding Salary (Decision Tree)")
plt.xlabel("Designation")
plt.ylabel("Salary")
plt.show()

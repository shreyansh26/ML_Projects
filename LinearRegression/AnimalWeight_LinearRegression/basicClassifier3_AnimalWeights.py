import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read data
data = pd.read_fwf('brain_body.txt')
x_values = data[['Brain']]
y_values = data[['Body']]


plt.scatter(x_values, y_values)
plt.xlabel('Body Weight')
plt.ylabel('Brain Weight')
plt.axis([0, 1000, 0, 1000])
plt.show()

# Train model on data
linearModel = LinearRegression()
linearModel.fit(x_values, y_values)

# Visualise result
plt.scatter(x_values, y_values)
plt.plot(x_values, linearModel.predict(x_values), color='green')
plt.show()
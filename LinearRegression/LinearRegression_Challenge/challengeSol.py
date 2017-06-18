import numpy as np 
import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Read data
data = pd.read_csv('challenge_dataset.csv')
#print(data.to_string(index=False))

x_values = data['Val_x'].values.reshape(-1,1)
y_values = data['Val_y']

# Train model on data
linearModel = LinearRegression()
linearModel.fit(x_values, y_values)

# Visualise result
plt.scatter(x_values, y_values)
plt.plot(x_values, linearModel.predict(x_values), color='green')
plt.show()

from sklearn import tree
import numpy as np

# Training Data
features = [[140, 1], [130 ,1], [150, 0], [170, 0]]   # 0 => Smooth, 1 => Bumpy
labels = [0, 0, 1, 1]     # 0 => Orange, 1 => Apple

my_dict = {0: 'Orange', 1: 'Apple'}
# Classifier => Dexision Tree
dec_tree = tree.DecisionTreeClassifier()

dec_tree = dec_tree.fit(features, labels)

predictions = dec_tree.predict([[150, 0], [170, 1]])

fruit_prediction = np.vectorize(my_dict.get)(predictions)

print(fruit_prediction)

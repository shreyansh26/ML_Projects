from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
import pydot
import numpy as np

iris = load_iris()
#print(iris.data.shape)
#print(iris.target.shape)

# Separate testing data
test_idx = [0, 50, 100]

data = np.column_stack((iris.data, iris.target))
np.random.shuffle(data)
#print(data[0:50])

iris.data = data[:, 0:4]
#print(iris.data)
iris.target = data[:, 4:5]
#print(iris.target)

# Training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# Testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Fit model
dec_tree = DecisionTreeClassifier()
dec_tree.fit(train_data, train_target)

print(test_target.reshape(1,3))
print(dec_tree.predict(test_data))


# Visualise data
dot_data = StringIO()
export_graphviz(dec_tree, out_file='irisTree.dot', feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, impurity=False)
'''
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("iris.pdf")
'''
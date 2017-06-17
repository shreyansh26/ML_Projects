from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.externals.six import StringIO
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import pydot
import numpy as np

iris = load_iris()

X = iris.data 
y = iris.target
#print(X[0:10])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# Decision Tree Classifier
my_classifier = DecisionTreeClassifier()
my_classifier.fit(X_train, y_train)

print("Decision Tree Classifier")
predictions = my_classifier.predict(X_test)
print(predictions)
#print(y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

print()

# KNeighbor Classifier
my_classifier2 = KNeighborsClassifier()
my_classifier2.fit(X_train, y_train)

print("KNeighbor Classifier")
predictions2 = my_classifier2.predict(X_test)
print(predictions2)
#print(y_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions2))

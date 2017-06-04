from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

import pandas as pd

#[height, weight, shoesize]
X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166,65,40], [190,90,47], [175,64,39], [177,70,40], [159,55,37],
	[171,75,42], [181,85,43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']


#Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X,Y)
prediction = decision_tree.predict([[190,70,43], [186,65, 39]])
acc_decisiontree = round(decision_tree.score(X, Y) * 100, 2)
print("Decision Tree: ")
print(prediction)
print(acc_decisiontree)
print()

#Support Vector Machines
svc = SVC()
svc.fit(X,Y)
prediction = svc.predict([[190,70,43], [186,65, 39]])
acc_svc = round(svc.score(X, Y) * 100, 2)
print("Support Vector Machines: ")
print(prediction)
print(acc_svc)
print()

#K-Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X, Y)
prediction = knn.predict([[190,70,43], [186,65, 39]])
acc_knn = round(knn.score(X, Y) * 100, 2)
print("KNN: ")
print(prediction)
print(acc_knn)
print()

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X, Y)
prediction = gaussian.predict([[190,70,43], [186,65, 39]])
acc_gaussian = round(gaussian.score(X, Y) * 100, 2)
print("Naive Bayes: ")
print(prediction)
print(acc_gaussian)
print()

#Perceptron
perceptron = Perceptron()
perceptron.fit(X, Y)
prediction = perceptron.predict([[190,70,43], [186,65, 39]])
acc_perceptron = round(perceptron.score(X, Y) * 100, 2)
print("Perceptron: ")
print(prediction)
print(acc_perceptron)
print()

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X, Y)
Y_pred = linear_svc.predict([[190,70,43], [186,65, 39]])
acc_linear_svc = round(linear_svc.score(X, Y) * 100, 2)
print("LinearSVC: ")
print(prediction)
print(acc_linear_svc)
print()

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X, Y)
prediction = sgd.predict([[190,70,43], [186,65, 39]])
acc_sgd = round(sgd.score(X, Y) * 100, 2)
print("Stochastic GD: ")
print(prediction)
print(acc_sgd)
print()

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X, Y)
prediction = random_forest.predict([[190,70,43], [186,65, 39]])
acc_random_forest = round(random_forest.score(X, Y) * 100, 2)
print("Random Forest: ")
print(prediction)
print(acc_sgd)
print()

models = pd.DataFrame({
        'Model': ['Support Vector Machines', 'KNN', 'Random Forest', 'Naive Bayes', 'Perceptron',
                  'LinearSVC', 'Stochastic Gradient Descent', 'Decision Tree'],
        'Score': [acc_svc, acc_knn, 
              acc_random_forest, acc_gaussian, acc_perceptron, acc_linear_svc,
              acc_sgd, acc_decisiontree]
    })
print(models.sort_values(by='Score', ascending=False))

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255

x_train, x_test = X[:60000], X[60000:]
y_train, y_test = Y[:60000], Y[60000:]

# activations = ['logistic', 'tanh', 'relu']
# solvers = ['lbfgs', 'sgd', 'adam']


l2 = [ 0.0001, 0.001, 0.01, 0.1]

layers = [(50, 50, 50), (150, 50, 50), (100, 50), (50, 100)]

j = 0

for x in [0.001, 0.01]:
    mlp = MLPClassifier(max_iter=200, activation='logistic', alpha=x, solver='adam').fit(x_train,  y_train)
    y_hat = mlp.predict(x_test)
    print(1 - accuracy_score(y_hat, y_test))


for y in [0.001, 0.01]:
    mlp = MLPClassifier(max_iter=200, activation='tanh', alpha=y, solver='lbfgs').fit(x_train,  y_train)
    y_hat = mlp.predict(x_test)
    print(1 - accuracy_score(y_hat, y_test))

for i in l2:
    print(layers[j])
    print(i)
    mlp = MLPClassifier(hidden_layer_sizes = layers[j], max_iter=300, activation='relu', alpha=i, solver='lbfgs').fit(x_train,  y_train)
    y_hat = mlp.predict(x_test)
    print(1 - accuracy_score(y_hat, y_test))
    j += 1

for i in l2:
    print(layers[j])
    print(i)
    mlp = MLPClassifier(hidden_layer_sizes = layers[j], max_iter=300, activation='tanh', alpha=i, solver='lbfgs').fit(x_train,  y_train)
    y_hat = mlp.predict(x_test)
    print(1 - accuracy_score(y_hat, y_test))
    j += 1
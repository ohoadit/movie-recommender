import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255

x_train, x_test = X[:20000], X[60000:]
y_train, y_test = Y[:20000], Y[60000:]

knn = KNeighborsClassifier(algorithm='ball_tree')
knn.fit(x_train, y_train)
y_hat = knn.predict(x_test)
print(1 - accuracy_score(y_test, y_hat))

knn_a = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=10)
knn_a.fit(x_train, y_train)
y_hat_a = knn_a.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_a))

knn_b = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=3)
knn_b.fit(x_train, y_train)
y_hat_b = knn_b.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_b))

knn_c = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=5, p=1)
knn_c.fit(x_train, y_train)
y_hat_c = knn_c.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_c))

knn_d = KNeighborsClassifier(algorithm='ball_tree', n_neighbors=5, leaf_size=20)
knn_d.fit(x_train, y_train)
y_hat_d = knn_d.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_d))


knn_e = KNeighborsClassifier(algorithm='kd_tree')
knn_e.fit(x_train, y_train)
y_hat_e = knn_e.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_e))

# knn_f = KNeighborsClassifier()
# knn_f.fit(x_train, y_train)
# y_hat_f = knn_f.predict(x_test)
# print(1 - accuracy_score(y_test, y_hat_f))

# knn_g = KNeighborsClassifier(algorithm='kd_tree', p=1)
# knn_g.fit(x_train, y_train)
# y_hat_g = knn_g.predict(x_test)
# print(1 - accuracy_score(y_test, y_hat_g))

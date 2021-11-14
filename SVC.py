import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

X, Y = fetch_openml('mnist_784', version=1, return_X_y=True)
X = X / 255

x_train, x_test = X[:60000], X[60000:]
y_train, y_test = Y[:60000], Y[60000:]

svc_poly = svm.SVC(kernel='poly', degree=4)
svc_poly.fit(x_train, y_train)
y_hat_poly = svc_poly.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_poly))

svc_poly_a = svm.SVC(kernel='poly', degree=4, gamma=1.0)
svc_poly_a.fit(x_train, y_train)
y_hat_poly_a = svc_poly_a.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_poly_a))

svc_poly_b = svm.SVC(kernel='poly', degree=4, gamma = 1.0, C=0.01, coef0 = 25.5)
svc_poly_b.fit(x_train, y_train)
y_hat_poly_b = svc_poly_b.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_poly_b))

svc_poly_c = svm.SVC(kernel='poly', degree=3, gamma=1.8, C=0.01)
svc_poly_c.fit(x_train, y_train)
y_hat_poly_c = svc_poly_c.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_poly_c))

svc_poly_d = svm.SVC(kernel='linear')
svc_poly_d.fit(x_train, y_train)
y_hat_poly_d = svc_poly_d.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_poly_d))


svc_rbf = svm.SVC(kernel='rbf')
svc_rbf.fit(x_train, y_train)
y_hat_rbf = svc_rbf.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_rbf))

svc_rbf_a = svm.SVC(kernel='rbf', C = 1.8)
svc_rbf_a.fit(x_train, y_train)
y_hat_rbf_a = svc_rbf_a.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_rbf_a))

svc_sigma = svm.SVC(kernel='sigmoid', C = 0.5)
svc_sigma.fit(x_train, y_train)
y_hat_sigma = svc_sigma.predict(x_test)
print(1 - accuracy_score(y_test, y_hat_sigma))

svc_sigma_a = svm.SVC(kernel='sigmoid', C = 0.5, gamma = 20.5)
svc_sigma_a.fit(x_train, y_train)
y_hat_sigma_a = svc_sigma_a.predict(x_test)

print(1 - accuracy_score(y_test, y_hat_sigma))
print(1 - accuracy_score(y_test, y_hat_sigma_a))
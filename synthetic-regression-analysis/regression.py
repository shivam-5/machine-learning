# regression.py
# parsons/2017-2-05
#
# A simple example using regression.
#
# This illustrates both using the linear regression implmentation that is
# built into scikit-learn and the function to create a regression problem.
#
# Code is based on:
#
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
# http://scikit-learn.org/stable/auto_examples/linear_model/plot_ransac.html

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

#
# Generate a regression problem:
#

feature_count = 3

# The main parameters of make-regression are the number of samples, the number
# of features (how many dimensions the problem has), and the amount of noise.
X, y = make_regression(n_samples=100, n_features=feature_count, noise=20)

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

#
# Solve the problem using the built-in regression model
#

regr = linear_model.LinearRegression() # A regression model object
regr.fit(X_train, y_train)             # Train the regression model

#
# Evaluate the model
#

# Data on how good the model is:
print("Mean squared error: %.2f"
      % np.mean((regr.predict(X_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(X_test, y_test))

M = X_train.shape[0]
N = X_test.shape[0]
X_train = np.c_[np.ones(M), X_train]
X_test = np.c_[np.ones(N), X_test]
w = np.zeros(feature_count + 1)

# iter = 30
# alpha = 0.003
# gradient = "stochastic"

iter = 1000
alpha = 0.009
gradient = "batch"

errors = []
for i in range(0, iter):
    if gradient == "stochastic":
        for ii in range(0, M):
            y_hat = np.dot(X_train[ii, :], np.transpose(w))
            error = y_train[ii] - y_hat
            w = w + alpha * error * X_train[ii, :]
    else:
        prediction = np.dot(X_train, w)
        error = np.subtract(y_train, prediction)
        w = w + alpha * np.dot(np.transpose(X_train), error) / M
    mean_error = np.mean((np.dot(X_train, w) - y_train) ** 2)
    errors.append(mean_error)
print("Mean squared error test(manual): %.2f" % np.mean((np.dot(X_test, w) - y_test) ** 2))
plt.plot(np.arange(iter), errors)
plt.show()

# Plotting training data, test data, and results.
# plt.scatter(X_train, y_train, color="black")
# plt.scatter(X_test, y_test, color="red")
# plt.scatter(X_test, regr.predict(X_test), color="blue")
#
# plt.show()





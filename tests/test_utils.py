import pandas as pd
from pandas import DataFrame
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import os
import utils
from sklearn.datasets import make_blobs
#import pandas

# digits = datasets.load_digits()
test_size = 0.20
val_size = 0.10
#gamma_val = [0.00001,0.0001,0.001,0.01]

# n_samples = len(digits.images)
# data = digits.images.reshape((n_samples, -1))

#TODO-1

n_samples_1 = 100
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
gen_data = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
gen_data.label = gen_data.label.map({0:-1, 1:1})

X, y = gen_data[['x','y']], gen_data.label
test_size = 0.20
val_size = 0.10

X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(X,y,0.2,0.1)
#clf.fit(X_train, y_train)

X_train=X_train.to_numpy()
y_train=y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X=X.to_numpy()

xtrain_len = len(X_train)
xtest_len = len(X_test)
xval_len = len(X_val)




train_sample = n_samples_1*0.7
test_sample = n_samples_1*0.2
val_sample = n_samples_1*0.1
def test_size():
    assert train_sample == xtrain_len
    assert test_sample == xtest_len
    assert val_sample ==xval_len

#TODO-2

n_samples_1 = 9
X, y = make_blobs(n_samples=100, centers=2, n_features=2)
gen_data = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
gen_data.label = gen_data.label.map({0:-1, 1:1})

X, y = gen_data[['x','y']], gen_data.label
test_size = 0.20
val_size = 0.10

X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(X,y,0.2,0.1)
#clf.fit(X_train, y_train)

X_train=X_train.to_numpy()
y_train=y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X=X.to_numpy()

xtrain_len = len(X_train)
xtest_len = len(X_test)
xval_len = len(X_val)

train_sample = n_samples_1*0.7
test_sample = n_samples_1*0.2
val_sample = n_samples_1*0.1

    #train_metrics = utils.train_model(clf,X_train,y_train)

def test_size():
    assert (train_sample)==xtrain_len
    assert (test_sample)==xtest_len
    assert (val_sample)==xval_len
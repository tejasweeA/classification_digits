import pandas as pd
from pandas import DataFrame
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import os
import utils
from sklearn.datasets import make_blobs
import math

test_size = 0.20
val_size = 0.10

#TODO-1

data_samples = [100,9]
for data_len in data_samples:
    X, y = make_blobs(n_samples=data_len, centers=2, n_features=2)
    gen_data = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
    gen_data.label = gen_data.label.map({0:-1, 1:1})

    X, y = gen_data[['x','y']], gen_data.label
    test_size = 0.20
    val_size = 0.10

    X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(X,y,test_size,val_size)
    xtrain_len = len(X_train)
    xtest_len = len(X_test)
    xval_len = len(X_val)
    x_len = len(X)

    train_sample = round(data_len*0.7)
    test_sample = round(data_len*0.2)
    val_sample = round(data_len*0.1)
    sum = train_sample + test_sample + val_sample
    def test_size():
        assert train_sample == xtrain_len
        assert test_sample == xtest_len
        assert val_sample ==xval_len
        assert sum == x_len
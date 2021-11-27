from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import data, color
from sklearn import tree
from skimage.transform import rescale
import numpy as np
import os
import joblib
import utils
import random

digits = datasets.load_digits()


test_size = 0.15

val_size = test_size


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
A=[]
count=0

print("\nHyperparameters:           RUN1                       RUN2                 RUN3                  Mean\n")
print("  C value    Gamma     Train  Dev   Test       Train  Dev  Test       Train  Dev  Test       Train  Dev  Test\n")

c_value=[0.001,10,1000]
gamma_val=[0.001,0.01,1]
for sp in range(0,3):
    c_val = random.choice(c_value)
    gmval=random.choice(gamma_val)
    train_acc=[]
    val_acc=[]
    test_acc=[]
    for i in range(0,3):

        X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(data,digits.target,test_size,val_size)

        clf = svm.SVC(gamma = gmval, C=c_val)
        # classify.append(clf)
        clf.fit(X_train, y_train)
        clf_values_train = utils.train_model(clf,X_train,y_train)
        clf_values_val = utils.train_model(clf,X_val,y_val)
        clf_values_test = utils.train_model(clf,X_test,y_test)
        print(end=" ")
        # print("\n{:.4f}\t{:.4f}\t{:.4f}".format(clf_values_train['acc'],clf_values_val['acc'],clf_values_test['acc'], end=" "))
        train_acc.append(clf_values_train['acc'])
        val_acc.append(clf_values_val['acc'])
        test_acc.append(clf_values_test['acc'])
    
    print("{:.3f}  {:.3f}   ".format(c_val,gmval),end=" ")
    for j in range(0,3):
        print("{:.4f} {:.4f} {:.4f}   ".format(train_acc[j],val_acc[j],test_acc[j]), end=" ")
    print("{:.4f} {:.4f} {:.4f}".format(np.mean(train_acc),np.mean(val_acc),np.mean(test_acc)))
    print("")
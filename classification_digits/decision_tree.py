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

digits = datasets.load_digits()

print("test_size:")
test_size = float(input())

val_size = test_size


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
A=[]
count=0

print("Split:\tSVM:\t\tGamma value\tDecision Tree:\t\tMax_Depth\n")

classifiers = ['SVM','Decision_Tree']
best_clf = []
acc_svm=0
acc_dt=0
acc_svm=[]
acc_dt=[]
# for sp in range(0,5):
#     split = []
for clfs in classifiers:
    classify = []
    clf_values =[]
    if clfs == 'SVM':
        hyperparameter = [0.00001,0.0001,0.001,0.01,0.1,1.0]
        for gmvalue in hyperparameter:
            clf = svm.SVC(gamma = gmvalue)
            classify.append(clf)

    else:
        hyperparameter = [4,5,6,8]
        for hp in hyperparameter:
            clf = tree.DecisionTreeClassifier(max_depth=hp)
            classify.append(clf)
    max_acc = 0
    best_hp = 0
    for clf, hp in zip(classify, hyperparameter):
        X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(data,digits.target,test_size,val_size)

        clf.fit(X_train, y_train)

        clf_values,pred = utils.train_model(clf,X_val,y_val)
        print(clf_values)
        #print("\n{}\t\t{:.17f}\t{}".format(hp,clf_values['acc'],clf_values['f1']), end=" ")
        if clf_values["acc"]>max_acc:
            max_acc = clf_values['acc']
            best_hp = hp
    if clfs == 'SVM':
        acc_svm.append(max_acc)
        print("\n \t{} \t{} \t".format(format(max_acc,'.16f'),best_hp),end='')
        models_dir= "../model_select_svm/valid_{}_gamma_{}".format((test_size/2),best_hp)
        os.mkdir(models_dir)
        joblib.dump(clf,os.path.join(models_dir,"model_info.joblib"))
    else:
        acc_dt.append(max_acc)
        print("{} \t{}".format(format(max_acc,'.16f'),best_hp))
        models_dir= "../model_select_decision/valid_{}_depth_{}".format((test_size/2),best_hp)
        os.mkdir(models_dir)
        joblib.dump(clf,os.path.join(models_dir,"model_info.joblib"))
    print("")

print("\n SVM:\n-----\n Mean:{}  Standard deviation:{}".format(np.mean(acc_svm),np.std(acc_svm)))
print("\n Decision Tree:\n----------\n Mean:{}  Standard deviation:{}".format(np.mean(acc_dt),np.std(acc_dt)))
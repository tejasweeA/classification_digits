from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from skimage import data
import numpy as np
import os
import seaborn as sns
import joblib
import utils
import pandas as pd



digits = datasets.load_digits()

test_size = 0.1

val_size = test_size


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
A=[]
count=0
train_split=0.8
print("Split:\tSVM Accuracy:\tGamma value")

models_data=[]

train_splits=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
hyperparameter = [0.00001,0.0001,0.001,0.01,0.1,1.0]
X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(data,digits.target,train_split)
best_clf = []
acc_svm=0
acc_dt=0
acc_svm=[]
acc_dt=[]
for train_sp in train_splits:
    classify=[]
    models_dump=[]
    max_acc = 0
    best_hp=0
    for gmvalue in hyperparameter:
        clf = svm.SVC(gamma = gmvalue)
        classify.append(clf)
        if train_sp != 1:
            train_x,_,_,train_y,_,_ = utils.split_dataset(X_train,y_train,train_sp)
        else:
            train_x,train_y = X_train,y_train
        clf.fit(train_x, train_y)
        clf_values = utils.train_model(clf,X_val,y_val)
        models_dir = "../model_select/valid_{}_gamma_{}".format(train_sp,gmvalue)
        os.mkdir(models_dir)
        joblib.dump(clf,os.path.join(models_dir,"model_info.joblib")) 
        if clf_values['acc']>max_acc:
            max_acc = clf_values['acc']
            best_hp = gmvalue
            models = {
                    "Train_split":train_sp*100,
                    "f1_val":clf_values['f1'],
                    "acc_val":clf_values['acc'],
                    "Gamma value":gmvalue
                }
        models_dump.append(models)
        acc_svm.append(max_acc)
    print("\n {} \t{} \t{} \t".format(train_sp,format(max_acc,'.16f'),best_hp),end='')

    best_val = max(models_dump,key=lambda x:x['f1_val'])
    best_model_dir = "../model_select/valid_{}_gamma_{}".format(train_sp,best_hp)
    clf = joblib.load(os.path.join(best_model_dir,"model_info.joblib"))

    clf_values_test = utils.train_model(clf,X_test,y_test)
    pred = clf.predict(X_test)
    acc_load = metrics.accuracy_score(y_pred=pred, y_true=y_test)
    f1_load = metrics.f1_score(y_pred=pred, y_true=y_test, average='macro')

    plt.figure()
    cm = confusion_matrix(y_test,pred)
    cm
    sns.heatmap(cm,annot=True)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    filename = "Confusion-Matrix for "+str(train_sp*100)+" train_data.png"
    plt.savefig(filename)
    del pred
    test_data = {
            "acc_test":acc_load,
            "f1_test":f1_load
        }
    best_val.update(test_data)
    models_data.append(best_val)
df = pd.DataFrame(models_data)
print("\n")
print(df)
plt.figure()
plt.plot(df['Train_split'],df['f1_test'])
plt.xlabel("Training split")
plt.ylabel("F1 Score of Test Data")
plt.title("Training split   Vs.  F1 score of Test Data")
plt.savefig("train_f1.png")
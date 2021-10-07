from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
import utils
import sys

digits = datasets.load_digits()


print("test_size:")
test_size = 0.15
val_size=0.15

print("initial gamma value:")
gmvalue = 0.00001


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
A=[]
count=0


print("Gamma Value:\tAccuracy:\t\tF1_Score:\n")
maxAcc=0
while gmvalue!=1000:
        arr=[]
        clf = svm.SVC(gamma=gmvalue)
   
  
        X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(data,digits.target,test_size,val_size)
        #X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(data, digits.target, test_size, valid_size
        clf.fit(X_train, y_train)

        clf_values = utils.train_model(clf,X_val,y_val)
    
        print("{}\t\t{:.17f}\t{}".format(gmvalue,clf_values['acc'],clf_values['f1']), end=" ")

        if clf_values['acc'] < 0.11:
            print("Low accuracy, not efficient")
            gmvalue=gmvalue*10
            continue


        if (clf_values['acc']>maxAcc):
            maxAcc=clf_values['acc']
            corr_gamma=gmvalue
            max_x_val = X_val
            max_y_val = y_val
        print("")

        models_dir = "../model_select/valid_{}_gamma_{}".format((test_size),gmvalue)
        os.mkdir(models_dir)
        joblib.dump(clf,os.path.join(models_dir,"model_info.joblib")) 
     
        gmvalue=gmvalue*10

best_model_dir = utils.model_selection(test_size,corr_gamma)
clf = joblib.load(os.path.join(best_model_dir,"model_info.joblib"))
pred = clf.predict(max_x_val)
acc_load = metrics.accuracy_score(y_pred=pred, y_true=max_y_val)
f1_load = metrics.f1_score(y_pred=pred, y_true=max_y_val, average='macro')

print("\nMax accuracy: {} is found for gamma value:{} with f1 score as {}\n".format(acc_load,corr_gamma,f1_load))

from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale
import numpy as np
import os
import joblib
import utils

digits = datasets.load_digits()

'''_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)'''

print("test_size:")
test_size = float(input())

print("initial gamma value:")
#gmvalue = 0.00001
gmvalue=float(input())



n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
A=[]
count=0


print("Gamma Value:\tAccuracy:\t\tF1_Score:\n")
maxAcc=0
while gmvalue!=1000:
        arr=[]
        clf = svm.SVC(gamma=gmvalue)
   
  
        X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(data,digits.target,test_size)
        
        clf.fit(X_train, y_train)

        clf_values = utils.train_model(clf,X_val,y_val)
        #predicted = clf.predict(X_test)

        #acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        #f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average='macro')
        #arr.append(gmvalue)
        #arr.append(acc)
        #arr.append(f1)
        print("{}\t\t{:.17f}\t{}".format(gmvalue,clf_values['acc'],clf_values['f1']), end=" ")

        if clf_values['acc'] < 0.11:
            print("Low accuracy, not efficient")
            gmvalue=gmvalue*10
            continue


        models_dir = "../model_select/valid_{}_gamma_{}".format((test_size/2),gmvalue)
        os.mkdir(models_dir)
        joblib.dump(clf,os.path.join(models_dir,"model_info.joblib")) 
        #A.append(arr)


        if (clf_values['acc']>maxAcc):
            maxAcc=clf_values['acc']
            corr_gamma=gmvalue
        print("")
        gmvalue=gmvalue*10

best_model_dir = "../model_select/valid_{}_gamma_{}".format((test_size/2),corr_gamma)
clf = joblib.load(os.path.join(best_model_dir,"model_info.joblib"))
pred = clf.predict(X_test)
acc_load = metrics.accuracy_score(y_pred=pred, y_true=y_test)
f1_load = metrics.f1_score(y_pred=pred, y_true=y_test, average='macro')

print("\nMax accuracy: {} is found for gamma value:{} with f1 score as {}\n".format(acc_load,corr_gamma,f1_load))
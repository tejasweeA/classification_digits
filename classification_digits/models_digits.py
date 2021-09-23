from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale
import numpy as np
import os
import joblib


digits = datasets.load_digits()

'''_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)'''


test_size = 0.3
# flatten the image
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
gmvalue = 0.00001
A=[]
count=0
print("Gamma Value:\tAccuracy:\t\tF1_Score:\n")
maxAcc=0
while gmvalue!=1000:
        arr=[]
        # Create a classifier: a support vector classifier
        clf = svm.SVC(gamma=gmvalue)

        # Split data into 50% train and 50% test subsets
        X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, random_state=1)

        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=1)
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)

        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)
        _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
        for ax, image, prediction in zip(axes, X_test, predicted):
               ax.set_axis_off()
               image = image.reshape(8, 8)
               ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
               ax.set_title(f'Prediction: {prediction}')

        ###############################################################################
        disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
        disp.figure_.suptitle("Confusion Matrix")
        # print(f"Confusion matrix:\n{disp.confusion_matrix}")

        acc = metrics.accuracy_score(y_pred=predicted, y_true=y_test)
        f1 = metrics.f1_score(y_pred=predicted, y_true=y_test, average='macro')
        arr.append(gmvalue)
        arr.append(acc)
        arr.append(f1)
        print("{}\t\t{:.17f}\t{}".format(gmvalue,acc,f1), end=" ")
        #print("gmvalue:\t Accuracy:\t F1_Score:")
        if acc < 0.11:
            print("Low accuracy, not efficient")
            gmvalue=gmvalue*10
            continue
        #####################################################################
        models_dir = "../model_select/valid_{}_gamma_{}".format((test_size/2),gmvalue)
        os.mkdir(models_dir)
        joblib.dump(clf,os.path.join(models_dir,"model_info.joblib")) 
        A.append(arr)
        count=count+1
        #valid_split = (y_test/X_test)*100
        if (acc>maxAcc):
            maxAcc=acc
            corr_gamma=gmvalue
        print("")
        gmvalue=gmvalue*10

#print("value of max split",valid_max)
best_model_dir = "../model_select/valid_{}_gamma_{}".format((test_size/2),corr_gamma)
clf = joblib.load(os.path.join(best_model_dir,"model_info.joblib"))
pred = clf.predict(X_test)
acc_load = metrics.accuracy_score(y_pred=pred, y_true=y_test)
f1_load = metrics.f1_score(y_pred=pred, y_true=y_test, average='macro')



print("\nMax accuracy: {} is found for gamma value:{} with f1 score as {}\n".format(acc_load,corr_gamma,f1_load))

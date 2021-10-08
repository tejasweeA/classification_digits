#from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from pandas import DataFrame

# test_size = 0.2
# val_size = 0.1

def split_dataset(data,target,test_size,val_size):
    test_size = test_size + val_size
    X_train, X_test_val, y_train, y_test_val = train_test_split(data, target, test_size= test_size, shuffle=False)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=val_size / (test_size), shuffle=False)

    return X_train, X_test, X_val, y_train,y_test,y_val


def model_selection(test_size,corr_gamma):
    best_model_dir = "../model_select/valid_{}_gamma_{}".format((test_size),corr_gamma)
    return best_model_dir



def train_model(clf,X,y):

    model_values = dict();
    predicted = clf.predict(X)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1_score = metrics.f1_score(y_pred=predicted, y_true=y, average='macro')
    model_values["acc"] = acc
    model_values["f1"] = f1_score

    return model_values



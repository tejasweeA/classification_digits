from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics


def split_dataset(data,target,train_data):
    test_data=0.1
    val_data=0.1

    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=train_data, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, train_size=(val_data/(val_data + test_data)), shuffle=False)

    return X_train, X_test, X_val, y_train,y_test,y_val

def train_model(clf,X,y):

    model_values = dict();
    predicted = clf.predict(X)
    acc = metrics.accuracy_score(y_pred=predicted, y_true=y)
    f1_score = metrics.f1_score(y_pred=predicted, y_true=y, average='macro')
    model_values["acc"] = acc
    model_values["f1"] = f1_score

    return model_values	



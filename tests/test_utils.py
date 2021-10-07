from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import os
import utils

digits = datasets.load_digits()
test_size = 0.15
val_size=0.15
gamma_val = [0.00001,0.0001,0.001,0.01]

n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

for gm_val in gamma_val:
    clf = svm.SVC(gamma=gm_val)
    
    X_train, X_test, X_val, y_train,y_test,y_val = utils.split_dataset(data,digits.target,test_size,val_size)
    clf.fit(X_train, y_train)

    train_metrics = utils.train_model(clf,X_train,y_train)


    # TODO-1

    def test_file_path():
        best_model_dir = utils.model_selection(test_size,gm_val)
        print(best_model_dir)
        assert os.path.exists(best_model_dir)

    #TODO-2

    def test_small_data_overfit_checking():
        assert train_metrics['acc']  > 0.11
        assert train_metrics['f1'] > 0.11
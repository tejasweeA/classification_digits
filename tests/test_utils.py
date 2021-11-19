from joblib import load
import os
import numpy as np


best_model = '/home/unix/mlops/classification_digits/model_select/valid_0.15_gamma_0.001/model_info.joblib'
clf = load(best_model)

image = ["0.0","0.0","0.0","11.999999999999982","13.000000000000004","5.000000000000021","8.881784197001265e-15","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999988","9.000000000000005","1.598721155460224e-14","0.0","0.0","0.0","2.9999999999999925","14.999999999999979","15.999999999999998","6.000000000000022","1.0658141036401509e-14","0.0","6.217248937900871e-15","6.999999999999987","14.99999999999998","15.999999999999996","16.0","2.0000000000000284","3.552713678800507e-15","0.0","5.5220263365470826e-30","6.21724893790087e-15","1.0000000000000113","15.99999999999998","16.0","3.000000000000022","5.32907051820075e-15","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000015","1.0658141036401498e-14","0.0","0.0","0.0","0.9999999999999989","15.99999999999998","16.0","6.000000000000018","1.0658141036401503e-14","0.0","0.0","0.0","0.0","10.999999999999986","15.999999999999993","10.00000000000001","1.7763568394002505e-14","0.0"]
image = np.array(image).reshape(1,-1)

def test_num0():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 0

def test_num1():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 1

def test_digit_num2():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 2

def test_num3():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 3

def test_num4():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 4

def test_num5():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 5

def test_num6():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 6

def test_num7():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 7

def test_num8():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 8

def test_num9():

    predicted = clf.predict(image)
    temp = int(predicted[0])
    assert temp == 9
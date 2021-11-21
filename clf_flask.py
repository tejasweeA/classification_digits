from flask import Flask
from flask import request
import numpy as np
import joblib


app = Flask(__name__)


best_model_path_svm = 'model_select_svm/valid_0.15_gamma_0.001/model_info.joblib'
best_model_path_decision = 'model_select_decision/valid_0.15_depth_8/model_info.joblib'
# gamma=0.001

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/svm_predict',methods=['POST'])
def predict():
    clf = joblib.load(best_model_path_svm)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print(str(predicted[0]))
    return str(predicted[0])

@app.route('/svm_decision_tree',methods=['POST'])
def predict_decision_tree():
    clf = joblib.load(best_model_path_decision)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print(str(predicted[0]))
    return str(predicted[0])

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=8000)
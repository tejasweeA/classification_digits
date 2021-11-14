from flask import Flask
from flask import request
import numpy as np
import joblib


app = Flask(__name__)


best_model_path = 'model_select/valid_0.15_gamma_0.01/model_info.joblib'

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/predict',methods=['POST'])
def predict():
    clf = joblib.load(best_model_path)
    input_json = request.json
    image = input_json['image']
    print(image)
    image = np.array(image).reshape(1, -1)
    predicted = clf.predict(image)
    print(str(predicted[0]))
    return str(predicted[0])
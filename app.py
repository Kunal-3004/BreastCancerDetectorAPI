import numpy as np
import pandas as pd
from flask import Flask, request
import pickle

app = Flask(__name__)
model = pickle.load(open('breast_cancer_detector.pkl', 'rb'))

@app.route('/')
def home():
    return "Breast Cancer Prediction API is running! Use POST /predict"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    features_name = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    
    try:
        input_features = [float(data[col]) for col in features_name]
    except KeyError as e:
        return {"error": f"Missing feature: {str(e)}"}, 400
    
    df = pd.DataFrame([input_features], columns=features_name)
    output = model.predict(df)
    
    res_val = "Breast Cancer" if output[0] == 0 else "No Breast Cancer"
    
    return {"prediction": res_val}


if __name__ == "__main__":
    app.run(debug=True)

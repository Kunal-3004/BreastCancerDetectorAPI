# 🧬 Breast Cancer Detector API

This project provides a **Flask-based REST API** to detect breast cancer using machine learning.  
The trained model (`breast_cancer_detector.pkl`) can classify whether a tumor is malignant or benign based on medical features.

---

## 📊 How the Model Was Built

1. **Data Preprocessing**
   - Collected the Breast Cancer dataset (from `sklearn.datasets` or CSV).
   - Cleaned and handled missing values.

2. **Feature Scaling**
   - Applied **standardization (scaling)** to ensure all features contribute equally.

3. **Exploratory Data Analysis (EDA)**
   - Plotted the **correlation matrix** to analyze relationships between features.
   - Selected the most relevant features.

4. **Model Training**
   - Trained multiple machine learning models:
     - Decision Tree  
     - Random Forest  
     - Support Vector Machine (SVM)  
     - K-Nearest Neighbors (KNN)  
     - Logistic Regression  

   - After comparison, **Logistic Regression** gave the **best performance**.

5. **Model Serialization**
   - Saved the trained Logistic Regression model as `breast_cancer_detector.pkl` using **pickle**.

---

## 🚀 API Setup

### 🔹 Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/Kunal-3004/BreastCancerDetectorAPI.git
   cd BreastCancerDetectorAPI
2.Create a virtual environment & install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # for Windows
   source .venv/bin/activate  # for Linux/Mac

   pip install -r requirements.txt

3.Run Flask server:
    ```bash
   python app.py

4.http://127.0.0.1:5000/predict
    ```bash
   This api is deployed on Render.You can test by using below url
   https://breastcancerdetectorapi.onrender.com/predict

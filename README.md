# 🩺 Diabetes Prediction using SVM

**A simple, reproducible machine learning project to predict diabetes (Positive/Negative) using the Pima Indians Diabetes Dataset and Support Vector Machine (SVM).**

---

## 🔎 Project Overview

This project builds a binary classification model that predicts whether a person has diabetes based on clinical measurements (e.g., glucose, BMI, age). The model uses **Support Vector Machine (SVM)** with appropriate preprocessing (feature scaling) to produce reliable results.

---

## 📚 Dataset

* **Name:** Pima Indians Diabetes Dataset (commonly available on UCI / Kaggle)
* **Rows:** 768 (typical)
* **Features:** 8 numeric features

  * Pregnancies
  * Glucose
  * BloodPressure
  * SkinThickness
  * Insulin
  * BMI
  * DiabetesPedigreeFunction
  * Age
* **Target:** `Outcome` (0 = Non-diabetic, 1 = Diabetic)

> Note: If you use a local copy, place it in `dataset/diabetes.csv` or update the path in the notebook/script.

---

## 🧭 Project Steps

1. Load dataset with pandas
2. Exploratory Data Analysis (EDA) and missing-value checks
3. Split data into features (`X`) and target (`Y`)
4. Train/test split (e.g., 80% train, 20% test) with stratification
5. Scale features using `StandardScaler` (important for SVM)
6. Train SVM classifier (e.g., `kernel='rbf'` or `kernel='linear'`)
7. Evaluate using accuracy, confusion matrix, precision, recall, F1-score, ROC-AUC
8. Save model (optional) and build a prediction function

---

## ⚙️ Requirements

Create a virtual environment and install dependencies:

```bash
pip install -r requirements.txt
```

**Example `requirements.txt`:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

---

## 🧪 Quick Usage (Notebook / Script)

Below is a short example showing the typical workflow used in the project.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load
data = pd.read_csv('dataset/diabetes.csv')
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
classifier = svm.SVC(kernel='rbf', probability=True)
classifier.fit(X_train, y_train)

# Predict & Evaluate
y_pred = classifier.predict(X_test)
print('Test Accuracy:', accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))

# Save model (optional)
import joblib
joblib.dump(classifier, 'models/svm_diabetes_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
```

---

## 📌 Example: Making a Single Prediction

```python
import numpy as np
import joblib

scaler = joblib.load('models/scaler.joblib')
model = joblib.load('models/svm_diabetes_model.joblib')

# example input: [Pregnancies, Glucose, BP, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
sample = np.array([[2, 120, 70, 20, 79, 25.5, 0.5, 30]])
sample_scaled = scaler.transform(sample)
pred = model.predict(sample_scaled)
print('Diabetic' if pred[0] == 1 else 'Non-diabetic')
```

---

## 📁 Recommended Repository Structure

```
├── dataset/
│   └── diabetes.csv
├── models/
│   └── svm_diabetes_model.joblib
├── notebooks/
│   └── diabetes_prediction.ipynb
├── src/
│   └── train.py
│   └── predict.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ✅ Evaluation Metrics (suggested)

* **Accuracy** — fraction of correct predictions
* **Precision** — useful when false positives are important
* **Recall (Sensitivity)** — important for medical diagnosis (catch positives)
* **F1-score** — harmonic mean of precision and recall
* **Confusion Matrix** — shows TP, TN, FP, FN
* **ROC-AUC** — separability of classes

---

## 🔧 Tips & Best Practices

* Always **scale** features before SVM (`StandardScaler`) to avoid dominated features
* Use **stratify=y** in `train_test_split` to preserve class balance
* Try `GridSearchCV` or `RandomizedSearchCV` to tune `C`, `gamma`, and kernel
* For imbalanced data consider class weights or resampling (SMOTE)
* Save the scaler along with the model to ensure consistent preprocessing

---

## ♻️ Future Improvements

* Compare with other models (Logistic Regression, Random Forest, XGBoost)
* Add cross-validation and hyperparameter tuning
* Build a small web UI using Streamlit or Flask
* Create an API with FastAPI for real-time predictions

---

## 📄 License

This project is released under the MIT License. Feel free to reuse and modify.

---

## 👤 Author

**Arif Azeem**
GitHub: [https://github.com/ArifAzeem786](https://github.com/ArifAzeem786)
LinkedIn: [https://linkedin.com/in/arif-azeem-7282782a3](https://linkedin.com/in/arif-azeem-7282782a3)

---

If you want, I can also:

* generate a `requirements.txt` file for you, or
* create a ready-to-upload `diabetes_prediction.ipynb` with cells for each step.

# ❤️ Heart Disease Prediction System

A Machine Learning-powered web application that predicts the likelihood of heart disease based on patient health parameters. The project uses data preprocessing, feature scaling, and a trained classification model to provide instant predictions through an interactive Streamlit interface.

---

## 📌 Project Overview

Heart disease remains one of the leading causes of death worldwide. Early prediction can help individuals seek timely medical attention and adopt preventive measures.

This project leverages Machine Learning techniques to analyze patient health data and predict the risk of heart disease based on multiple medical attributes.

The application is deployed using Streamlit, allowing users to input health parameters and receive real-time predictions.

---

## 🚀 Features

- Interactive web interface built with Streamlit
- Real-time heart disease risk prediction
- Data preprocessing and feature scaling
- Trained Machine Learning classification model
- User-friendly input forms
- Instant prediction results
- Lightweight and easy to deploy

---

## 🛠️ Tech Stack

### Programming Language
- Python

### Libraries Used
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Pickle

### Machine Learning
- Classification Model
- Feature Scaling
- Data Preprocessing

---

## 📊 Dataset Information

The project uses a Heart Disease Dataset containing patient medical records and health indicators.

### Features Used

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol Level
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate Achieved
- Exercise Induced Angina
- ST Depression
- Slope of Peak Exercise ST Segment
- Number of Major Vessels
- Thalassemia

### Target Variable

- Presence or absence of heart disease

---

## 📂 Project Structure

```text
heart-disease-prediction/
│
├── app.py
├── train_model.py
├── heart_model.pkl
├── scaler.pkl
├── Heart_Disease_Dataset.xlsx
├── Clinical_Report.pdf
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/dishaprabhakar2006-blip/heart-disease-prediction.git
```

### Navigate to the Project Directory

```bash
cd heart-disease-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

---

## 🌐 Live Demo

Try the deployed application here:

https://heart-disease-prediction-frsk2ze9xyzowviudve5tz.streamlit.app/

---

## 🔄 Workflow

1. Collect patient health data.
2. Preprocess and clean the dataset.
3. Scale numerical features.
4. Train the machine learning model.
5. Save the trained model using Pickle.
6. Deploy the model using Streamlit.
7. Accept user inputs and generate predictions.

---

## 🎯 Future Enhancements

- Improve model accuracy using advanced algorithms
- Add visual analytics dashboard
- Integrate electronic health record support
- Deploy on cloud infrastructure
- Include risk score interpretation and recommendations

---

## 👩‍💻 Author

**Disha P**

GitHub:
https://github.com/dishaprabhakar2006-blip

LinkedIn:
https://www.linkedin.com/in/disha-p-46668232b/

---

## 📜 License

This project is developed for educational and learning purposes.

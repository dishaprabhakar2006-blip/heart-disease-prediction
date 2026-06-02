import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Load Dataset
print("Loading dataset...")
df = pd.read_excel("heart diesease dataset.xlsx")

# Encode categorical columns
print("Encoding categorical columns...")
categorical_cols = [
    'sex',
    'chest_pain_type',
    'fasting_blood_sugar',
    'rest_ecg',
    'exercise_induced_angina',
    'slope',
    'vessels_colored_by_flourosopy',
    'thalassemia'
]

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Features and Target
X = df.drop("target", axis=1)
y = df["target"]

# Scaling
print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train model
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save model and scaler
joblib.dump(model, "heart_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")

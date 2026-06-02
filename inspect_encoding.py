import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_excel("heart diesease dataset.xlsx")

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

print("Column Categorical Mappings:")
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = df[col].astype(str) # Let's make sure it's read as string if not already
    encoded = le.fit_transform(df[col])
    print(f"\n--- {col} ---")
    mapping = dict(zip(le.classes_, range(len(le.classes_))))
    for cls, val in mapping.items():
        print(f"  {cls} -> {val}")

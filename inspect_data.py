import pandas as pd

df = pd.read_excel("heart diesease dataset.xlsx")
print("Dataset Shape:", df.shape)
print("\nColumns and Types:")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget counts:")
print(df['target'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

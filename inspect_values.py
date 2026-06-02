import pandas as pd

df = pd.read_excel("heart diesease dataset.xlsx")
for col in df.columns:
    print(f"\n--- {col} ---")
    print(df[col].unique())

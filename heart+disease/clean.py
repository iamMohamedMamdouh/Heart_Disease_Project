import pandas as pd
import numpy as np

file_path = 'processed.cleveland.data'
df = pd.read_csv(file_path, header=None)
df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

df.replace('?', np.nan, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)

df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

df.to_csv('heart_disease.csv', index=False)

print("✅ Data processed and saved as 'heart_disease.csv'")

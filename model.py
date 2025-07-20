import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('dataset/hair_health_dataset.csv')

# ✅ FIX: Fill missing values in Hair Issues
df['Hair Issues'] = df['Hair Issues'].fillna('')

# Convert "Hair Issues" to issue count
df['Hair Issues Count'] = df['Hair Issues'].apply(lambda x: len(x.split(',')) if x else 0)

# Encode categorical columns
le_cols = ['Stress Level', 'Pollution Exposure', 'Hair Coloring Frequency',
           'Hair Care Budget', 'Genetic/Hormonal Wellness']

label_encoders = {}
for col in le_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature matrix
X = df[['Stress Level', 'Sleep Hours', 'Water Intake (L)', 'Pollution Exposure',
        'Hair Coloring Frequency', 'Hair Issues Count', 'Hair Care Budget', 'Genetic/Hormonal Wellness']]

# Targets
y_score = df['Hair Health Score']
le_risk = LabelEncoder()
y_risk = le_risk.fit_transform(df['Hair Risk Level'])

# Train-test split
X_train, X_test, y_train_score, y_test_score = train_test_split(X, y_score, test_size=0.2, random_state=42)
_, _, y_train_risk, y_test_risk = train_test_split(X, y_risk, test_size=0.2, random_state=42)

# Train models
reg_model = RandomForestRegressor()
reg_model.fit(X_train, y_train_score)

clf_model = RandomForestClassifier()
clf_model.fit(X_train, y_train_risk)

# Save models
joblib.dump(reg_model, 'score_model.pkl')
joblib.dump(clf_model, 'risk_model.pkl')
joblib.dump(le_risk, 'risk_encoder.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

print("✅ Models trained and saved successfully.")
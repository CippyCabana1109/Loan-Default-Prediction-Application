import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# === Load dataset ===
df = pd.read_csv(r"C:\Users\CYPRIAN\Documents\DATASET 1\Loan_default.csv")

# === Encode target ===
df['Default'] = df['Default'].map({'No': 0, 'Yes': 1}) if df['Default'].dtype == 'object' else df['Default']

# === Identify feature types ===
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).drop("Default", axis=1, errors='ignore').columns

# === Preprocessing pipeline ===
cat_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("label_enc", LabelEncoder())  # will handle manually below
])

num_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# === Label encode categoricals manually ===
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# === Fill missing numerics ===
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# === Define X and y ===
X = df.drop('Default', axis=1)
y = df['Default']

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Apply SMOTE ===
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# === Hyperparameter tuning (RandomizedSearch) ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt'],
    'bootstrap': [True]
}

rfc = RandomForestClassifier(random_state=42)
search = RandomizedSearchCV(rfc, param_grid, n_iter=5, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
search.fit(X_train_res, y_train_res)
best_model = search.best_estimator_

# === Evaluate ===
preds = best_model.predict(X_test)
probs = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
print("ROC AUC:", roc_auc_score(y_test, probs))

# === Save model ===
dump(best_model, 'loan_default_model.joblib')

# === Save results to CSV ===
results = X_test.copy()
results['actual_default'] = y_test.values
results['predicted_default'] = preds
results['prob_default'] = probs
results.to_csv('prediction2_results.csv', index=False)

# === SHAP Explainability ===
# Ensure all features are numeric
X_shap = X_test.copy()
X_shap = X_shap.apply(pd.to_numeric, errors='coerce')
X_shap.fillna(0, inplace=True)

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap)

# Save SHAP summary plot
shap.summary_plot(shap_values[1], X_shap, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches='tight')
plt.close()

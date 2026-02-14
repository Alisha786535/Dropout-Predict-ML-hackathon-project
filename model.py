# ============================================================
# STUDENT DROPOUT EARLY WARNING SYSTEM (FIXED VERSION)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

import joblib
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# STYLE
# ============================================================

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

print("="*60)
print("STUDENT DROPOUT EARLY WARNING SYSTEM")
print("="*60)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("Data.csv")   # change if needed

print(f"\nDataset Shape: {df.shape}")

# ============================================================
# DATA CLEANING
# ============================================================

df.columns = df.columns.str.strip().str.lower()

# Rename columns correctly
column_mapping = {
    'raisedhands': 'raised_hands',
    'visitedresources': 'visited_resources',
    'announcementsview': 'announcements_view',
    'discussion': 'discussion_posts',
    'parentansweringsurvey': 'parentanswerssurvey'
}

df.rename(columns=column_mapping, inplace=True)

print("\nColumns:")
print(df.columns.tolist())

# ============================================================
# TARGET CREATION (FIXED)
# ============================================================

if 'class' not in df.columns:
    raise ValueError("Target column 'class' not found!")

print("\nUnique classes:", df['class'].unique())

# CORRECT mapping for xAPI dataset
target_mapping = {
    'L': 1,   # dropout risk
    'M': 0,
    'H': 0
}

df['dropout_risk'] = df['class'].map(target_mapping)

print("\nTarget Distribution:")
print(df['dropout_risk'].value_counts())

# Safety check
if df['dropout_risk'].nunique() < 2:
    raise ValueError("❌ Only one class found! Check target mapping.")

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df['engagement_score'] = (
    df['raised_hands'] * 0.3 +
    df['visited_resources'] * 0.3 +
    df['announcements_view'] * 0.2 +
    df['discussion_posts'] * 0.2
) / 100

df['low_engagement'] = (df['engagement_score'] < 0.3).astype(int)

# Parent involvement
if 'parentanswerssurvey' in df.columns:
    df['parent_involvement'] = (
        df['parentanswerssurvey']
        .str.lower()
        .map({'yes': 2, 'no': 0})
        .fillna(1)
    )
else:
    df['parent_involvement'] = 1

df['early_semester_proxy'] = (
    (df['raised_hands'] > 50) &
    (df['visited_resources'] > 50) &
    (df['announcements_view'] > 20)
).astype(int)

print("\nFeature engineering complete.")

# ============================================================
# FEATURE SELECTION
# ============================================================

features = [
    'gender','nationality','placeofbirth','stageid','gradeid',
    'sectionid','relation','parentschoolsatisfaction',
    'studentabsencedays','topic','semester',
    'raised_hands','visited_resources','announcements_view',
    'discussion_posts','engagement_score','low_engagement',
    'parent_involvement','early_semester_proxy'
]

features = [f for f in features if f in df.columns]

X = df[features]
y = df['dropout_risk']

# ============================================================
# PREPROCESSING
# ============================================================

cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(
        drop='first',
        sparse_output=False,
        handle_unknown='ignore'
    ))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols)
])

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# ============================================================
# MODEL
# ============================================================

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42
    ))
])

print("\nTraining model...")
rf_pipeline.fit(X_train, y_train)

y_pred = rf_pipeline.predict(X_test)
y_pred_proba = rf_pipeline.predict_proba(X_test)[:,1]

print("Model trained successfully.")

# ============================================================
# EVALUATION
# ============================================================

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC: {roc_auc:.3f}")

cv_scores = cross_val_score(
    rf_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='roc_auc'
)

print(f"\nCV ROC-AUC: {cv_scores.mean():.3f}")

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

ohe_features = (
    rf_pipeline.named_steps['preprocessor']
    .named_transformers_['cat']
    .named_steps['onehot']
    .get_feature_names_out(cat_cols)
)

feature_names = num_cols + list(ohe_features)

importances = rf_pipeline.named_steps['classifier'].feature_importances_

min_len = min(len(feature_names), len(importances))

feature_df = pd.DataFrame({
    "feature": feature_names[:min_len],
    "importance": importances[:min_len]
}).sort_values("importance", ascending=False)

print("\nTop Features:")
print(feature_df.head(10))

# Plot
plt.figure(figsize=(10,6))
top = feature_df.head(10)
plt.barh(top["feature"], top["importance"])
plt.gca().invert_yaxis()
plt.title("Top Feature Importance")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()

# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(rf_pipeline, "student_dropout_model.joblib")
print("\nModel saved successfully.")

# ============================================================
# CREATE PREDICTIONS FILE
# ============================================================

all_probs = rf_pipeline.predict_proba(X)[:,1]

def risk_label(x):
    if x < 0.3:
        return "Low"
    elif x < 0.6:
        return "Medium"
    else:
        return "High"

predictions_df = pd.DataFrame({
    "student_id":[f"STU{str(i+1000).zfill(4)}" for i in range(len(df))],
    "risk_score": all_probs,
    "risk_label":[risk_label(x) for x in all_probs],
    "predicted_dropout": (all_probs >= 0.5).astype(int)
})

predictions_df.to_csv("student_predictions.csv", index=False)

print("\nPrediction file saved: student_predictions.csv")

print("\n🎉 TRAINING COMPLETE 🎉")

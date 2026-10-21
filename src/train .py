
from pathlib import Path
import glob
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PLOT_DIR = ROOT / "outputs" / "plots"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "outputs" / "reports"

for folder in [PLOT_DIR, MODEL_DIR, REPORT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 1. Locate and load the dataset
csv_files = glob.glob(str(ROOT / "data" / "raw" / "**" / "*.csv"),
                      recursive=True)

if not csv_files:
    raise FileNotFoundError(
        "No CSV found in data/raw. Extract the dataset ZIP first."
    )

csv_path = Path(csv_files[0])
print(f"\nDataset: {csv_path}")
df = pd.read_csv(csv_path, low_memory=False)

if "label" not in df.columns:
    raise ValueError(f"Target column 'label' missing. Found: {df.columns.tolist()}")

print("Dataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts(dropna=False))

# 2. Basic data cleaning
df = df.dropna(subset=["label"]).copy()
df = df.drop_duplicates().reset_index(drop=True)

# Remove identifiers and raw text fields.
# The initial benchmark uses numeric dataset features.
excluded = ["label", "FILENAME", "URL", "Domain", "TLD", "Title"]
X = df.drop(columns=excluded, errors="ignore")
X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)

# Drop columns that contain no usable information
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

y = df["label"].astype(int)

if y.nunique() != 2:
    raise ValueError(f"Expected binary labels, found: {sorted(y.unique())}")

feature_names = X.columns.tolist()

print("\nRows after cleaning:", len(df))
print("Numeric features:", len(feature_names))
print("Features:", feature_names)

# 3. Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 4. Define models
models = {
    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
}

# Add XGBoost only if it is installed
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ))
    ])
    print("\nXGBoost is available.")
except ImportError:
    print("\nXGBoost is not installed; continuing with two models.")

# 5. Train, evaluate, and compare
results = []
trained_models = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    scores = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "ROC_AUC": roc_auc_score(y_test, probabilities)
    }

    results.append(scores)
    trained_models[name] = model

    print(pd.Series(scores).to_string())
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

print("\n=== MODEL COMPARISON ===")
print(results_df.to_string(index=False))

# Select the best model by F1 score
best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]
best_predictions = best_model.predict(X_test)
best_probabilities = best_model.predict_proba(X_test)[:, 1]

joblib.dump(
    {
        "model": best_model,
        "features": feature_names,
        "target": "label",
        "best_model_name": best_name
    },
    MODEL_DIR / "phishguard_model.joblib"
)

with open(MODEL_DIR / "model_config.json", "w", encoding="utf-8") as f:
    json.dump({
        "model": best_name,
        "features": feature_names,
        "target": "label",
        "random_state": 42
    }, f, indent=2)

# 6. Visualization 1: Class distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="label")
plt.title("Phishing Dataset: Label Distribution")
plt.xlabel("Dataset label")
plt.ylabel("Number of records")
plt.tight_layout()
plt.savefig(PLOT_DIR / "01_label_distribution.png", dpi=150)
plt.close()

# 7. Visualization 2: Feature correlation heatmap
corr_cols = X.columns[:25].tolist()
plt.figure(figsize=(14, 10))
sns.heatmap(
    X[corr_cols].corr(),
    cmap="coolwarm",
    center=0,
    xticklabels=True,
    yticklabels=True
)
plt.title("Feature Correlation Heatmap (First 25 Numeric Features)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "02_correlation_heatmap.png", dpi=150)
plt.close()

# 8. Visualization 3: Confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, best_predictions, cmap="Blues"
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig(PLOT_DIR / "03_confusion_matrix.png", dpi=150)
plt.close()

# 9. Visualization 4: ROC curves
plt.figure(figsize=(8, 6))
for name, model in trained_models.items():
    RocCurveDisplay.from_estimator(
        model, X_test, y_test, name=name, ax=plt.gca()
    )
plt.plot([0, 1], [0, 1], "k--", label="Random baseline")
plt.title("ROC Curve Comparison")
plt.tight_layout()
plt.savefig(PLOT_DIR / "04_roc_curves.png", dpi=150)
plt.close()

# 10. Visualization 5: Feature importance
classifier = best_model.named_steps["classifier"]

if hasattr(classifier, "feature_importances_"):
    importance = classifier.feature_importances_
elif hasattr(classifier, "coef_"):
    importance = np.abs(classifier.coef_[0])
else:
    importance = None

if importance is not None:
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(20)

    importance_df.to_csv(
        REPORT_DIR / "feature_importance.csv", index=False
    )

    plt.figure(figsize=(10, 8))
    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )
    plt.title(f"Top 20 Feature Importances - {best_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_feature_importance.png", dpi=150)
    plt.close()

print("\nTraining completed.")
print("Best model:", best_name)
print("Saved model:", MODEL_DIR / "phishguard_model.joblib")
print("Comparison:", REPORT_DIR / "model_comparison.csv")
print("Plots:", PLOT_DIR)

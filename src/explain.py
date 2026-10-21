
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "phishguard_model.joblib"
PLOT_DIR = ROOT / "outputs" / "plots"
REPORT_DIR = ROOT / "outputs" / "reports"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run python src/train.py first.")

saved = joblib.load(MODEL_PATH)
model = saved["model"]
feature_names = saved["features"]
model_name = saved["best_model_name"]

print("Loaded model:", model_name)

# Load the original dataset
import glob

files = glob.glob(str(ROOT / "data" / "raw" / "**" / "*.csv"),
                  recursive=True)

if not files:
    raise FileNotFoundError("Dataset CSV not found in data/raw.")

df = pd.read_csv(files[0], low_memory=False)
df = df.dropna(subset=["label"]).drop_duplicates().reset_index(drop=True)

X = df[feature_names].replace([np.inf, -np.inf], np.nan)
X = X.sample(n=min(100, len(X)), random_state=42)

# Impute missing values using the fitted preprocessing step
imputer = model.named_steps["imputer"]
X_imputed = pd.DataFrame(
    imputer.transform(X),
    columns=feature_names,
    index=X.index
)

classifier = model.named_steps["classifier"]

# Explain model predictions
try:
    import shap

    print("Generating SHAP explanations...")

    # Explain the classifier after preprocessing
    if hasattr(classifier, "feature_importances_"):
        explainer = shap.TreeExplainer(classifier)
        values = explainer.shap_values(X_imputed)

        if isinstance(values, list):
            values = values[-1]
        elif getattr(values, "ndim", 0) == 3:
            values = values[:, :, -1]

        shap.summary_plot(
            values,
            X_imputed,
            feature_names=feature_names,
            show=False,
            max_display=20
        )

        plt.tight_layout()
        plt.savefig(
            PLOT_DIR / "06_shap_summary.png",
            dpi=150,
            bbox_inches="tight"
        )
        plt.close()

        mean_abs = np.abs(values).mean(axis=0)
        explanation = pd.DataFrame({
            "Feature": feature_names,
            "MeanAbsoluteSHAP": mean_abs
        }).sort_values("MeanAbsoluteSHAP", ascending=False)

        explanation.to_csv(
            REPORT_DIR / "shap_feature_importance.csv",
            index=False
        )

        print("\nTop features influencing predictions:")
        print(explanation.head(10).to_string(index=False))

    else:
        print(
            "The selected model is not tree-based. "
            "Using permutation importance instead."
        )
        raise ImportError("Use fallback")

except Exception as exc:
    print("SHAP explanation unavailable:", str(exc))
    print("Creating model-agnostic permutation importance instead.")

    from sklearn.inspection import permutation_importance

    # Use a small sample to keep execution practical
    sample_count = min(1000, len(df))
    sample = df.sample(n=sample_count, random_state=42)

    X_sample = sample[feature_names].replace(
        [np.inf, -np.inf], np.nan
    )
    y_sample = sample["label"].astype(int)

    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring="f1",
        n_repeats=3,
        random_state=42,
        n_jobs=1
    )

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Importance": result.importances_mean
    }).sort_values("Importance", ascending=False)

    explanation.to_csv(
        REPORT_DIR / "permutation_importance.csv",
        index=False
    )

    top = explanation.head(20).sort_values("Importance")

    plt.figure(figsize=(10, 8))
    plt.barh(top["Feature"], top["Importance"])
    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean decrease in F1 score")
    plt.tight_layout()
    plt.savefig(
        PLOT_DIR / "06_permutation_importance.png",
        dpi=150
    )
    plt.close()

print("\nExplainability analysis finished.")
print("Plots saved in:", PLOT_DIR)
print("Reports saved in:", REPORT_DIR)

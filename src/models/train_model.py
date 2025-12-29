from __future__ import annotations

from pathlib import Path
import json
from typing import List, Dict, Any

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# ----------------------------
# Paths (project-root safe)
# ----------------------------
ROOT = Path(__file__).resolve().parents[2]          # .../digitaltwin
DATA_PATH = ROOT / "data" / "ai4i2020.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "pdm_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "train_metrics.json"


# ----------------------------
# Dataset conventions
# ----------------------------
TARGET_CANDIDATES = ["Machine failure", "machine_failure", "Target", "target"]

# AI4I 2020 often includes failure-mode columns.
# If your target is overall machine failure, these can become target leakage.
LEAKAGE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]

# Identifiers are not predictive in a generalizable sense.
ID_COLS = ["UDI", "Product ID"]


def detect_target_column(df: pd.DataFrame) -> str:
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"Target column not found. Expected one of {TARGET_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def build_pipeline(num_cols: List[str], cat_cols: List[str]) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        n_estimators=400,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


def train_and_save(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    y_col = detect_target_column(df)

    # Drop identifiers
    drop_cols = [c for c in ID_COLS if c in df.columns]

    # Drop leakage columns for overall failure targets
    y_norm = y_col.lower().replace(" ", "_")
    if y_norm in ("machine_failure", "target"):
        drop_cols += [c for c in LEAKAGE_COLS if c in df.columns]
    if y_col == "Machine failure":
        drop_cols += [c for c in LEAKAGE_COLS if c in df.columns]

    # Build X/y
    X = df.drop(columns=[y_col] + drop_cols, errors="ignore")
    y = df[y_col].astype(int)

    # Column typing
    num_cols = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    if len(num_cols) == 0 and len(cat_cols) == 0:
        raise ValueError("No features left after dropping columns. Check dataset columns.")

    pipe = build_pipeline(num_cols=num_cols, cat_cols=cat_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if y.nunique() == 2 else None,
    )

    pipe.fit(X_train, y_train)

    # Metrics on hold-out split
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else float("nan")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred).tolist()

    # Save bundle (model artifact)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": pipe,
        "y_col": y_col,
        "drop_cols": drop_cols,
        "feature_cols": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    joblib.dump(bundle, MODEL_PATH)

    # Save metrics (for thesis Chapter 6)
    metrics = {
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "report": report,
        "y_col": y_col,
        "drop_cols": drop_cols,
        "feature_cols": X.columns.tolist(),
        "data_path": str(DATA_PATH),
        "model_path": str(MODEL_PATH),
        "test_size": test_size,
        "random_state": random_state,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


if __name__ == "__main__":
    m = train_and_save()
    print("Saved model to:", m["model_path"])
    print("Saved metrics to:", str(METRICS_PATH))
    print("ROC-AUC:", m["roc_auc"])

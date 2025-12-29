from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

# Project root: .../digitaltwin
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "pdm_pipeline.joblib"


def load_bundle() -> dict:
    """Load the trained model bundle (pipeline + metadata)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}. Train first:\n"
            f"  python -m src.models.train_model"
        )
    return joblib.load(MODEL_PATH)


def predict_one(bundle: dict, row: dict) -> float:
    """
    Predict failure probability for one sensor snapshot.
    row: dict with feature columns (same as during training, minus target).
    """
    pipe = bundle["pipeline"]
    feature_cols = bundle["feature_cols"]

    # keep only known features (robust against extra keys)
    row = {k: row[k] for k in row.keys() if k in feature_cols}
    X = pd.DataFrame([row])

    proba = float(pipe.predict_proba(X)[:, 1][0])
    return proba

from __future__ import annotations

from pathlib import Path
import sys
import time
import json
from datetime import datetime
from src.models.inference import load_bundle, predict_one

import streamlit as st
import pandas as pd


# ----------------------------
# Path setup (Windows + Streamlit safe)
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # .../digitaltwin
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.core.data_loader import load_data
from src.models.inference import load_bundle, predict_one


# ----------------------------
# Constants
# ----------------------------
TARGET_CANDIDATES = ["Machine failure", "machine_failure", "Target", "target"]
LOG_DIR = ROOT_DIR / "logs"
LOG_PATH = LOG_DIR / "predictions.csv"
METRICS_PATH = ROOT_DIR / "models" / "train_metrics.json"


# ----------------------------
# Helpers
# ----------------------------
def detect_target_column(df: pd.DataFrame) -> str | None:
    return next((c for c in TARGET_CANDIDATES if c in df.columns), None)


def append_log_row(row_dict: dict) -> None:
    """
    Append one row to logs/predictions.csv (create file with header if missing).
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    df_row = pd.DataFrame([row_dict])
    write_header = not LOG_PATH.exists()
    df_row.to_csv(LOG_PATH, mode="a", index=False, header=write_header)


@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource
def get_model_bundle():
    return load_bundle()


def load_train_metrics() -> dict | None:
    if METRICS_PATH.exists():
        try:
            return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def compute_feature_importance(bundle: dict, top_k: int = 10) -> pd.DataFrame | None:
    """
    Return top-k feature importances if classifier supports it (RandomForest does).
    We also try to get preprocessor feature names.
    """
    pipe = bundle.get("pipeline", None)
    if pipe is None:
        return None

    clf = pipe.named_steps.get("clf", None)
    pre = pipe.named_steps.get("pre", None)
    if clf is None or not hasattr(clf, "feature_importances_"):
        return None

    importances = clf.feature_importances_

    # Try to recover transformed feature names (best-effort)
    names = None
    if pre is not None and hasattr(pre, "get_feature_names_out"):
        try:
            names = pre.get_feature_names_out()
        except Exception:
            names = None

    if names is None or len(names) != len(importances):
        # fallback: generic names
        names = [f"f{i}" for i in range(len(importances))]

    fi = (
        pd.DataFrame({"feature": names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    return fi


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Digital Twin – Predictive Maintenance", layout="wide")
st.title("AI-Enhanced Digital Twin Prototype")
st.caption("AI4I 2020 predictive maintenance dataset")


df = get_data()
y_col = detect_target_column(df)

# Sidebar controls (global)
st.sidebar.header("Controls")
delay = st.sidebar.slider("Streaming delay (seconds)", 0.01, 1.0, 0.15, 0.01)
threshold = st.sidebar.slider("Alert threshold", 0.05, 0.95, 0.50, 0.01)
max_points = st.sidebar.slider("Plot window (points)", 20, 500, 120, 10)
log_enabled = st.sidebar.checkbox("Enable logging to logs/predictions.csv", value=True)
log_input_snapshot = st.sidebar.checkbox("Also log selected sensor values", value=False)

tabs = st.tabs(["Runtime (Digital Twin)", "Evaluation"])


# ============================
# Tab 1: Runtime
# ============================
with tabs[0]:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("Dataset preview")
        st.dataframe(df.head(30), use_container_width=True)

        st.markdown("**Dataset columns (for documentation):**")
        st.code(", ".join(df.columns), language="text")

    with right:
        st.subheader("Digital Twin runtime")

        # Load model bundle once (cached)
        try:
            bundle = get_model_bundle()
            st.success("Model loaded: models/pdm_pipeline.joblib")
        except Exception as e:
            st.error(str(e))
            st.info("Train the model first:\n\npython -m src.models.train_model")
            st.stop()

        # Prepare simulation frame: remove target if present
        sim_df = df.copy()
        if y_col is not None:
            sim_df = sim_df.drop(columns=[y_col])

        # Session state
        st.session_state.setdefault("running", False)
        st.session_state.setdefault("history", [])  # list[dict]
        st.session_state.setdefault("idx", 0)

        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        if c1.button("Start", use_container_width=True):
            st.session_state.running = True
        if c2.button("Stop", use_container_width=True):
            st.session_state.running = False
        if c3.button("Reset", use_container_width=True):
            st.session_state.running = False
            st.session_state.history = []
            st.session_state.idx = 0
            # optional: keep logs; do not delete automatically

        # Optional: clear log file button
        if c4.button("Clear log file", use_container_width=True):
            if LOG_PATH.exists():
                LOG_PATH.unlink()
            st.toast("Log file cleared (if it existed).", icon=None)

        placeholder_metrics = st.empty()
        placeholder_chart = st.empty()
        placeholder_table = st.empty()

        # --- One step per rerun (Streamlit-safe streaming) ---
        if st.session_state.running and st.session_state.idx < len(sim_df):
            row_series = sim_df.iloc[st.session_state.idx]
            row_dict = row_series.to_dict()

            proba = float(predict_one(bundle, row_dict))
            status = "ALERT" if proba >= threshold else "Normal"

            entry = {
                "timestamp_iso": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                "step": int(st.session_state.idx),
                "failure_proba": float(proba),
                "threshold": float(threshold),
                "status": status,
            }

            # Optional: log a small input snapshot (avoid huge logs)
            if log_input_snapshot:
                for k in ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]:
                    if k in row_dict:
                        entry[k] = row_dict[k]

            st.session_state.history.append(entry)
            st.session_state.history = st.session_state.history[-max_points:]
            st.session_state.idx += 1

            if log_enabled:
                append_log_row(entry)

            time.sleep(delay)
            st.rerun()

        # Render history
        hist_df = pd.DataFrame(st.session_state.history)

        with placeholder_metrics.container():
            m1, m2, m3, m4 = st.columns(4)
            if len(hist_df) > 0:
                m1.metric("Failure probability", f"{hist_df.iloc[-1]['failure_proba']:.3f}")
                m2.metric("Status", str(hist_df.iloc[-1]["status"]))
            else:
                m1.metric("Failure probability", "-")
                m2.metric("Status", "-")

            m3.metric("Step", str(st.session_state.idx))
            m4.metric("Log file", str(LOG_PATH.relative_to(ROOT_DIR)) if log_enabled else "Disabled")

        if len(hist_df) > 0:
            placeholder_chart.line_chart(hist_df.set_index("step")[["failure_proba"]], height=240)
            placeholder_table.dataframe(hist_df.tail(15), use_container_width=True)
        else:
            placeholder_chart.info("Press Start to begin streaming.")

        # Quick download of logs (if exists)
        if LOG_PATH.exists():
            st.download_button(
                "Download runtime log (CSV)",
                data=LOG_PATH.read_bytes(),
                file_name="predictions.csv",
                mime="text/csv",
            )


# ============================
# Tab 2: Evaluation
# ============================
with tabs[1]:
    st.subheader("Training metrics (from models/train_metrics.json)")

    metrics = load_train_metrics()
    if metrics is None:
        st.warning(
            "No training metrics found yet. Train the model first:\n\n"
            "python -m src.models.train_model\n\n"
            "This will create models/train_metrics.json."
        )
    else:
        # High-level KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{metrics.get('roc_auc', float('nan')):.3f}" if metrics.get("roc_auc") == metrics.get("roc_auc") else "n/a")
        c2.metric("Target column", str(metrics.get("y_col", "-")))
        c3.metric("Dropped cols", str(len(metrics.get("drop_cols", []))))

        st.markdown("**Dropped columns (IDs/leakage):**")
        st.code(", ".join(metrics.get("drop_cols", [])) or "-", language="text")

        # Confusion matrix
        cm = metrics.get("confusion_matrix", None)
        if cm is not None:
            st.markdown("**Confusion matrix (test split):**")
            st.dataframe(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]), use_container_width=True)

        # Classification report
        report = metrics.get("report", None)
        if report is not None:
            st.markdown("**Classification report (test split):**")
            rep_df = pd.DataFrame(report).T
            st.dataframe(rep_df, use_container_width=True)

    st.divider()
    st.subheader("Model explainability (feature importance)")

    # Load model bundle and show feature importances
    try:
        bundle = get_model_bundle()
        fi = compute_feature_importance(bundle, top_k=12)
        if fi is None:
            st.info("Feature importance not available for this classifier.")
        else:
            st.dataframe(fi, use_container_width=True)
            st.bar_chart(fi.set_index("feature")["importance"])
    except Exception as e:
        st.warning(f"Could not load model for explainability: {e}")

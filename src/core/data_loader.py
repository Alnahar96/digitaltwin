from pathlib import Path
import pandas as pd

# Project root: DIGITALTWIN
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "ai4i2020.csv"

def load_data() -> pd.DataFrame:
    """Load the AI4I 2020 predictive maintenance dataset."""
    df = pd.read_csv(DATA_PATH)
    return df

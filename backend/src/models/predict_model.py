import warnings
warnings.filterwarnings("ignore")

import gzip
import io
import os
import pickle
from typing import Iterable, Optional

import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(ROOT_DIR, "models", "multisim_xgb.pkl.gz")

def load_model(filepath: str):
    with gzip.open(filepath, "rb") as f:
        return pickle.load(f)

def _load_dataframe_from_bytes(file_content: bytes, filename: Optional[str]) -> pd.DataFrame:
    name = (filename or "").lower()
    buffer = io.BytesIO(file_content)

    if name.endswith(".csv"):
        return pd.read_csv(buffer)
    elif name.endswith(".parquet"):
        return pd.read_parquet(buffer)
    elif name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, Parquet, or Excel.")

def main(file_content: Optional[bytes] = None, filename: Optional[str] = None) -> Iterable:
    if file_content:
        X_test = _load_dataframe_from_bytes(file_content, filename)
    else:
        X_test = pd.read_parquet(os.path.join(ROOT_DIR, "data", "processed", "multisim_dataset_fe3.parquet"))
        X_test = X_test.drop(columns=["target"]).sample(100)

    model = load_model(MODEL_PATH)
    y_pred = model.predict(X_test)

    try:
        return y_pred.tolist()
    except TypeError:
        return [p for p in y_pred]

if __name__ == "__main__":
    preds = main()
    print(f"Generated {len(preds)} predictions")

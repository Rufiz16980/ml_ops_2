import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os

# Root of the repo (two levels up from this file)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Input and output paths
RAW_PATH = os.path.join(ROOT_DIR, "data", "raw", "multisim_dataset.parquet")
PROCESSED_PATH = os.path.join(ROOT_DIR, "data", "processed", "multisim_dataset_fe3.parquet")

COLUMNS = [
    "trf",
    "age",
    "gndr",
    "tenure",
    "age_dev",
    "dev_man",
    "device_os_name",
    "dev_num",
    "is_dualsim",
    "is_featurephone",
    "is_smartphone",
    "simcard_type",
    "region",
    "target",
]

def main():
    print(f"Reading parquet: {RAW_PATH}")
    df = pq.read_table(RAW_PATH, columns=COLUMNS)
    df = df.slice(0, 1_000_000).to_pandas()

    # Fix numeric columns
    numeric_cols = ["age", "tenure", "age_dev", "dev_num"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Cap unrealistic values
    df.loc[df["age"] > 100, "age"] = np.nan

    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_parquet(PROCESSED_PATH)
    print(f"✅ Saved processed dataset to {PROCESSED_PATH} with shape {df.shape}")


if __name__ == "__main__":
    main()

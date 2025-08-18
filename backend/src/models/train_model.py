import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import gzip
import pickletools
import pandas as pd

from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# Root of the repo
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
PROCESSED_PATH = os.path.join(ROOT_DIR, "data", "processed", "multisim_dataset_fe3.parquet")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
MODEL_FILE = "multisim_xgb.pkl.gz"

os.makedirs(MODELS_DIR, exist_ok=True)

def save_model(filename: str, model: object):
    file_path = os.path.join(MODELS_DIR, filename)
    with gzip.open(file_path, "wb") as f:
        pickled = pickle.dumps(model)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)

def main():
    print(f"Loading processed dataset: {PROCESSED_PATH}")
    df = pd.read_parquet(PROCESSED_PATH)

    target = "target"
    X = df.drop(columns=[target])
    y = df[target]

    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocess = ColumnTransformer(
        [
            ("num", "passthrough", numeric_cols),
            ("cat", CatBoostEncoder(random_state=0), categorical_cols),
        ]
    )

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric="logloss",
    )

    model_pipeline = Pipeline([("preproc", preprocess), ("xgb", xgb)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training model...")
    model_pipeline.fit(X_train, y_train)

    # Save model
    save_model(MODEL_FILE, model_pipeline)
    print(f"✅ Model saved to {os.path.join(MODELS_DIR, MODEL_FILE)}")

if __name__ == "__main__":
    main()

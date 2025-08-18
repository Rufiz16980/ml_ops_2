import os
import io
import gzip
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# -------------------------
# FastAPI initialization
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Model
# -------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "multisim_xgb.pkl.gz")

try:
    with gzip.open(MODEL_PATH, "rb") as f:
        model = joblib.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Could not load model: {e}")

# -------------------------
# Prediction endpoint
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read uploaded file into DataFrame
        if file.filename.endswith(".csv"):
            contents = await file.read()
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith(".parquet"):
            contents = await file.read()
            df = pd.read_parquet(io.BytesIO(contents))
        elif file.filename.endswith(".xlsx"):
            contents = await file.read()
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Check if DataFrame is empty
        if df.empty:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        # Run predictions
        predictions = model.predict(df)

        # Return results as list
        return {"predictions": predictions.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import streamlit as st
import pandas as pd
import requests
from io import BytesIO

st.set_page_config(page_title="MLOps Example", layout="wide")


def load_df_from_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame | None:
    """Load CSV, Excel, or Parquet bytes into DataFrame (for preview)."""
    try:
        bio = BytesIO(file_bytes)
        if filename.lower().endswith(".csv"):
            return pd.read_csv(bio)
        elif filename.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(bio)
        elif filename.lower().endswith(".parquet"):   # 👈 Parquet support
            return pd.read_parquet(bio)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or Parquet.")
            return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def detect_mime(filename: str) -> str:
    """Detect MIME type from filename extension."""
    name = (filename or "").lower()
    if name.endswith(".csv"):
        return "text/csv"
    if name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if name.endswith(".xls"):
        return "application/vnd.ms-excel"
    if name.endswith(".parquet"):   # 👈 Parquet support
        return "application/octet-stream"  # generic binary
    return "application/octet-stream"


# ---------------- Streamlit App ----------------
st.title("MLOps Example: Model Inference App")

uploaded = st.file_uploader(
    "Choose a CSV, Excel, or Parquet file",
    type=["csv", "xlsx", "xls", "parquet"],   # 👈 Parquet allowed
    help="Supported: .csv, .xlsx, .xls, .parquet",
)

if uploaded:
    st.subheader("Preview of uploaded data")
    df_preview = load_df_from_bytes(uploaded.getvalue(), uploaded.name)
    if df_preview is not None:
        st.dataframe(df_preview.head())

    if st.button("Run Inference", type="primary"):
        with st.spinner("Sending file to backend..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), detect_mime(uploaded.name))}
            try:
                # ✅ Use local backend when running without Docker
                response = requests.post("http://127.0.0.1:8000/predict", files=files)

                if response.status_code == 200:
                    result_json = response.json()

                    # ✅ Extract nested predictions
                    if "data" in result_json and "predictions" in result_json["data"]:
                        preds = result_json["data"]["predictions"]
                        st.subheader("Predictions")
                        st.dataframe(pd.DataFrame(preds, columns=["Prediction"]))

                        # Download button for results
                        csv_data = pd.DataFrame(preds, columns=["Prediction"]).to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv_data,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
                    else:
                        st.error("Backend did not return predictions.")
                else:
                    st.error(f"Backend error: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")

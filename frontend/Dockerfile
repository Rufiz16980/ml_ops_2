FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update -y && \
    apt-get install --no-install-recommends -y gcc && \
    rm -rf /var/lib/apt/lists/*

# Copy frontend code
COPY frontend/ .

# Install deps
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

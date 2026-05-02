FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    numpy pandas scikit-learn xgboost imbalanced-learn scikit-optimize \
    statsmodels scipy matplotlib seaborn \
    mlflow dagshub \
    streamlit plotly \
    typer python-dotenv loguru openpyxl pytest flit_core

COPY src/ src/
COPY app/ app/
COPY cli.py .
COPY experiments/ experiments/
COPY data/ data/

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]

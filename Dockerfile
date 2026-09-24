FROM python:3.11-slim

WORKDIR /app

# System libraries RDKit's Chem.Draw needs (same list used for Streamlit
# Community Cloud, see packages.txt)
COPY packages.txt .
RUN apt-get update \
    && xargs -a packages.txt apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py features.py solubility.py uncertainty.py ./
COPY drug_solubility_pipeline.joblib train_fingerprints.pkl uncertainty_calibration.json ./

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

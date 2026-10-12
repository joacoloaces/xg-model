FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# xgboost/catboost necesitan OpenMP en Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Solo lo necesario para servir el modelo
COPY src/__init__.py src/features.py src/preprocessing.py src/ensemble.py src/
COPY api/ api/
COPY models/xg_model.joblib models/xg_model.joblib

RUN useradd --create-home app && chown -R app:app /app
USER app

ENV MODEL_VERSION=1.0.0 \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.getenv('PORT','8000'))"

CMD ["sh", "-c", "uvicorn api.server:app --host 0.0.0.0 --port ${PORT}"]

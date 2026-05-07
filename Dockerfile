FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.11-slim

WORKDIR /app

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY api/ ./api/
COPY mlruns/ ./mlruns/
COPY params.yaml .

ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV MODEL_URI=file:///app/mlruns/1/models/m-daf3607be85a4404ae5a60a1913f3b14/artifacts

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/')" || exit 1

CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

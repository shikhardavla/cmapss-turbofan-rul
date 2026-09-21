FROM python:3.11-slim

WORKDIR /app

# CPU-only torch wheel -- much smaller than the default CUDA build.
# Inference here is single-sample/low-latency, so no GPU is needed to serve it.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# Pipeline package, config, and trained artifacts, baked into the image
# for a self-contained deployable. For frequently-retrained models, mount
# models/saved/ as a volume instead of COPYing it.
COPY src/ /app/src/
COPY config.yaml /app/config.yaml
COPY models/saved/ /app/models/saved/
COPY api/ /app/api/

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
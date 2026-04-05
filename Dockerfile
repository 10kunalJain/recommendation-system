# Multi-stage build for H&M Recommendation System
# Stage 1: Training  |  Stage 2: Serving (slim)

# ---- Base ----
FROM python:3.12-slim AS base
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Training ----
FROM base AS trainer
COPY . .
ENV OPENBLAS_NUM_THREADS=1
RUN python -c "from src.utils.config import CONFIG; print('Config loaded:', list(CONFIG.keys()))"
CMD ["python", "train.py"]

# ---- Serving (production) ----
FROM base AS server
COPY src/ src/
COPY configs/ configs/
COPY serve.py .
COPY artifacts/ artifacts/
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
CMD ["python", "serve.py", "--host", "0.0.0.0", "--port", "8000"]

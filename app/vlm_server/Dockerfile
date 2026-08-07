# CUDA-enabled base with torch preinstalled, so pip install here doesn't need
# to resolve a separate CUDA toolkit.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /srv

# requirements.txt installs deepseek_vl straight from GitHub (pip needs git
# on PATH to clone it) — the base image doesn't have git installed.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

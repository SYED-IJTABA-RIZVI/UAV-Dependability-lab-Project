FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /srv

# opencv-python (pulled in transitively by ultralytics) needs these X11/GL
# shared libraries at import time even though this service never opens a
# GUI window — the base image is minimal and doesn't have them.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

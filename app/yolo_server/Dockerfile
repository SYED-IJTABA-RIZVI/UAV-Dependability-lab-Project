FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /srv

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# ultralytics transitively installs opencv-python, which needs X11/GL shared
# libraries this minimal base image doesn't have (ImportError: libxcb.so.1 /
# libGL.so.1 at import time), even though this service never opens a GUI
# window. Force-reinstall the headless build afterward so it deterministically
# overwrites those files — avoids an `apt-get install` layer entirely (slow,
# hits the network for a package index update on every cache miss).
RUN pip install --no-cache-dir --force-reinstall --no-deps opencv-python-headless>=4.8

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

# CUDA-enabled base with torch preinstalled, so pip install here doesn't need
# to resolve a separate CUDA toolkit.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Docker isn't a TTY, so Python block-buffers stdout by default — both our
# own print()s and huggingface_hub's download progress output would only
# show up in `docker compose logs` in delayed chunks instead of live.
ENV PYTHONUNBUFFERED=1

WORKDIR /srv

# deepseek_vl has no PyPI release, only a GitHub repo — `pip install
# git+https://...` would need a `git` binary in the image (an apt-get layer,
# slow: hits the network for a package index refresh). ADD can fetch a URL
# directly with no such binary needed, so grab a tarball snapshot instead
# and pip-install from the extracted folder.
ADD https://github.com/deepseek-ai/DeepSeek-VL/archive/refs/heads/main.tar.gz /tmp/deepseek-vl.tar.gz
RUN mkdir -p /tmp/deepseek-vl && tar -xzf /tmp/deepseek-vl.tar.gz -C /tmp/deepseek-vl --strip-components=1 \
    && pip install --no-cache-dir /tmp/deepseek-vl \
    && rm -rf /tmp/deepseek-vl /tmp/deepseek-vl.tar.gz

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

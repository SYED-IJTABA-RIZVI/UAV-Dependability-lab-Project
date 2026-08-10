FROM python:3.12-slim

# Docker isn't a TTY, so Python block-buffers stdout by default — logs would
# only show up in `docker compose logs` / Docker Desktop's log viewer in
# delayed chunks instead of live.
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0", "--server.port=8501"]

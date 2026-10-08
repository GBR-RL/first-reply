FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HOME=/models/hf FIRST_REPLY_ROOT=/app
WORKDIR /app

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# Editable install: data and result paths resolve relative to /app.
RUN pip install -e ".[ml,service]"

# Data (prepared tickets, chunks, embedding cache) is mounted at /app/data.
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=600s --retries=60 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "first_reply.service.app:build", "--factory", "--host", "0.0.0.0", "--port", "8000"]

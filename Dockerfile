FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    WEB_CONCURRENCY=1 \
    GUNICORN_TIMEOUT=60 \
    GUNICORN_PRELOAD=0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends graphviz \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "PRELOAD_FLAG=''; if [ \"${GUNICORN_PRELOAD:-0}\" = \"1\" ] || [ \"${GUNICORN_PRELOAD:-0}\" = \"true\" ] || [ \"${GUNICORN_PRELOAD:-0}\" = \"TRUE\" ]; then PRELOAD_FLAG='--preload'; fi; exec gunicorn ${PRELOAD_FLAG} --timeout ${GUNICORN_TIMEOUT:-60} -w ${WEB_CONCURRENCY:-1} --bind 0.0.0.0:${PORT:-8080} dashboard:app"]

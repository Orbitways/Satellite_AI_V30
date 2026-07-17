FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY deploy/requirements-backend.txt /tmp/requirements-backend.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements-backend.txt

COPY . /app

RUN chmod +x /app/scripts/*.sh \
    && mkdir -p /app/data /app/backups

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3).read()" || exit 1

CMD ["bash", "scripts/start_backend_with_cdm_sync.sh"]

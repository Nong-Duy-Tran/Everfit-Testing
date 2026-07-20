FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Source lives under src/, so PYTHONPATH points there and `app.*` imports resolve.
ENV PYTHONPATH=/app/src

COPY src/ ./src/
COPY knowledge-base/ ./knowledge-base/
COPY sample-data/ ./sample-data/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

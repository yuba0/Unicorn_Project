FROM python:3.10-slim

WORKDIR /app

# Dépendances
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Code API et modèles
COPY api /app/api
COPY models /app/models

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

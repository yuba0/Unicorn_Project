FROM python:3.10-slim

WORKDIR /app

# Dépendances
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Code Dashboard et données processed
COPY dashboard.py /app/
COPY data/processed /app/data/processed

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]

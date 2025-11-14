FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src
    
WORKDIR /app

# copy and install dependencies
COPY requirements.txt .
RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

# copy project sources
COPY . .

EXPOSE 8000

# run the FastAPI app; adjust module path if your FastAPI app is in a different module
CMD ["fastapi", "dev", "server.py"]
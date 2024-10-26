FROM python:3.10-slim

WORKDIR /app
COPY . /app

# Install build-essential and other system dependencies for Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 3000
CMD ["python3", "./index.py"]

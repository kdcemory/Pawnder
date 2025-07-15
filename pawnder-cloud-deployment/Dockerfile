FROM python:3.9-slim

# Install system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY saved_model/ ./saved_model/

EXPOSE 8080
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]

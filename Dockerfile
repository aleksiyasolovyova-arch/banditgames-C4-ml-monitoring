FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6-dev \
    libpng-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY src/ /app/src/
COPY monitor.py .

# Create directories
RUN mkdir -p /workspace/datasets /workspace/tensorboard_logs

# Default command
CMD ["python", "monitor.py"]
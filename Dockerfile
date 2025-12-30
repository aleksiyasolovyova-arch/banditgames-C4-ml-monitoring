FROM python:3.10

WORKDIR /app

# System libraries (runtime only, no compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch explicitly
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY src/ /app/src/
COPY monitor.py .

# Create runtime directories
RUN mkdir -p /workspace/datasets /workspace/tensorboard_logs

# Default command
CMD ["python", "monitor.py"]

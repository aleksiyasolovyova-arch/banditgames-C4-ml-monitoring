FROM python:3.11-slim

LABEL maintainer="your-team@example.com"
LABEL service="connect4-ml-monitor"
LABEL version="1.0.0"

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy monitoring code
COPY performance_monitor.py .
COPY monitor.py .

# Reports directory (mounted as volume)
RUN mkdir -p /workspace/reports
VOLUME /workspace/reports

# Default command: run monitor
CMD ["python", "monitor.py"]

FROM python:3.11-slim

# System deps for OpenCV and PDF processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY app/ app/
COPY scripts/ scripts/
COPY tests/ tests/
COPY pyproject.toml .
COPY yolov8n.pt .

ENV PYTHONUNBUFFERED=1 \
    YOLO_VERBOSE=False \
    TRANSFORMERS_VERBOSITY=error \
    HF_HUB_DISABLE_PROGRESS_BARS=1

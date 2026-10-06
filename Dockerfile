# Dockerfile
# AMD Kria KV260 - Audio KWS Accelerator Container
# Compatible with both x86_64 (development) and linux/arm64 (Kria KV260 board)

FROM python:3.10-slim

# Avoid interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Install essential system libraries (audio codecs, C runtime, curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libportaudio2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory inside container
WORKDIR /app

# Copy dependency definition and install Python packages
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application source code, models, audio test samples, scripts, and FPGA hardware package
COPY pipeline/ /app/pipeline/
COPY board/ /app/board/
COPY benchmarks/ /app/benchmarks/
COPY models/ /app/models/
COPY data/test_inputs/ /app/data/test_inputs/
COPY scripts/ /app/scripts/
COPY hls/ /app/hls/
COPY vivado/ /app/vivado/
COPY CONFIG_C_HARDWARE_PACKAGE.md /app/CONFIG_C_HARDWARE_PACKAGE.md
COPY README.md /app/README.md

# Expose the Web Dashboard port
EXPOSE 8080

# Health check to ensure the web dashboard is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Default command: launch the interactive Web Dashboard
CMD ["python3", "board/app/web_ui.py"]

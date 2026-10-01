#!/usr/bin/env bash
# scripts/quantize.sh
# Step 9 — Docker wrapper to run Vitis AI quantization inside xilinx/vitis-ai-pytorch-cpu
#
# Usage:
#   bash scripts/quantize.sh [small|medium|large|gru]

set -euo pipefail

VARIANT=${1:-medium}
DOCKER_IMAGE="xilinx/vitis-ai-pytorch-cpu:latest"

echo "================================================================="
echo " Vitis AI Quantization (vai_q_pytorch) for DS-CNN (${VARIANT})"
echo "================================================================="

# Check if Docker is available
if ! command -v docker &>/dev/null; then
    echo "[!] ERROR: Docker is not installed or not in PATH."
    echo "    Please install Docker Desktop and pull ${DOCKER_IMAGE}"
    exit 1
fi

echo "[1/2] Running Calibration (mode=calib)..."
docker run --rm -v "$(pwd):/workspace" -w /workspace "${DOCKER_IMAGE}" \
    bash -c "conda activate vitis-ai-pytorch && python scripts/quantize.py --variant ${VARIANT} --quant_mode calib --subset_len 200"

echo "[2/2] Running Evaluation & XModel Export (mode=test --deploy)..."
docker run --rm -v "$(pwd):/workspace" -w /workspace "${DOCKER_IMAGE}" \
    bash -c "conda activate vitis-ai-pytorch && python scripts/quantize.py --variant ${VARIANT} --quant_mode test --subset_len 100 --deploy"

echo "[OK] Quantization complete. Quantized xmodel generated in models/quantized/"

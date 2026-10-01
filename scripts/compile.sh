#!/usr/bin/env bash
# scripts/compile.sh
# Step 10 — Compile quantized xmodel for Kria KV260 DPUCZDX8G using vai_c_xir
#
# Usage:
#   bash scripts/compile.sh [small|medium|large|gru] [B3136|B4096]

set -euo pipefail

VARIANT=${1:-medium}
ARCH_TYPE=${2:-B3136}
DOCKER_IMAGE="xilinx/vitis-ai-pytorch-cpu:latest"

ARCH_JSON="/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json"
NET_NAME="dscnn_${VARIANT}"
XMODEL_IN="./models/quantized/${NET_NAME}_int.xmodel"
OUT_DIR="./models/compiled"

echo "================================================================="
echo " Vitis AI Compilation (vai_c_xir) for KV260 DPU (${ARCH_TYPE})"
echo " Target Net: ${NET_NAME}"
echo "================================================================="

if ! command -v docker &>/dev/null; then
    echo "[!] ERROR: Docker is not installed or not in PATH."
    exit 1
fi

docker run --rm -v "$(pwd):/workspace" -w /workspace "${DOCKER_IMAGE}" \
    bash -c "conda activate vitis-ai-pytorch && \
    vai_c_xir \
        -x ${XMODEL_IN} \
        -a ${ARCH_JSON} \
        -o ${OUT_DIR} \
        -n ${NET_NAME}"

echo "[OK] Compilation complete. Compiled model: ${OUT_DIR}/${NET_NAME}.xmodel"
echo "     Inspecting subgraphs..."
docker run --rm -v "$(pwd):/workspace" -w /workspace "${DOCKER_IMAGE}" \
    bash -c "conda activate vitis-ai-pytorch && python scripts/inspect_subgraphs.py --xmodel ${OUT_DIR}/${NET_NAME}.xmodel"

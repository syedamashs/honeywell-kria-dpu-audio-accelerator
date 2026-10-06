#!/usr/bin/env bash
# docker_build.sh - Helper script for building the KV260 Docker image
set -euo pipefail

DOCKER_USER="syedamash07"
IMAGE_NAME="kria-kv260-kws"
TAG="latest"
FULL_IMAGE="${DOCKER_USER}/${IMAGE_NAME}:${TAG}"

echo "===================================================================="
echo " AMD Kria KV260 Docker Image Builder"
echo "===================================================================="

# Check Docker CLI
if ! command -v docker &>/dev/null; then
    echo "[!] Error: Docker is not installed or not in PATH."
    exit 1
fi

echo "[1/3] Building Docker image (${FULL_IMAGE})..."
docker build -t "${IMAGE_NAME}:${TAG}" -t "${FULL_IMAGE}" .

echo ""
echo "[2/3] Build completed successfully!"
echo ""
echo "[3/3] Next steps:"
echo "  1. Test locally:"
echo "     docker run -p 8080:8080 ${FULL_IMAGE}"
echo ""
echo "  2. Save offline .tar archive (for transfer to board without internet):"
echo "     docker save -o kv260_kws_image.tar ${FULL_IMAGE}"
echo ""
echo "  3. Push to Docker Hub:"
echo "     docker push ${FULL_IMAGE}"
echo "===================================================================="

#!/usr/bin/env bash
# scripts/setup_env.sh
# Step 2 — Create the kws_dpu Conda environment (Linux / WSL / macOS)
#
# Usage: bash scripts/setup_env.sh
#
# On Windows (native PowerShell) use: scripts\setup_env.bat

set -euo pipefail

echo "=== KWS-DPU Environment Setup ==="
echo ""

# Check conda
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found. Install Miniconda:"
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[1/4] Creating environment from environment.yml ..."
conda env create -f environment.yml || conda env update -f environment.yml --prune

echo ""
echo "[2/4] Activating environment ..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kws_dpu

echo ""
echo "[3/4] Verifying key packages ..."
python -c "import torch;        print('  torch       :', torch.__version__)"
python -c "import torchaudio;   print('  torchaudio  :', torchaudio.__version__)"
python -c "import librosa;      print('  librosa     :', librosa.__version__)"
python -c "import onnxruntime;  print('  onnxruntime :', onnxruntime.__version__)"
python -c "import numpy;        print('  numpy       :', numpy.__version__)"

echo ""
echo "[4/4] Running smoke tests (preprocessing + GEMM reference) ..."
python -m pytest pipeline/tests/ -v --tb=short

echo ""
echo "[OK] Setup complete. Activate with: conda activate kws_dpu"

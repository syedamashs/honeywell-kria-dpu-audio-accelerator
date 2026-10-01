@echo off
REM scripts/setup_env.bat
REM Step 2 — Create the kws_dpu Conda environment on Windows
REM
REM Usage: scripts\setup_env.bat
REM
REM Requirements: Miniconda or Anaconda installed and on PATH.
REM Download Miniconda: https://docs.conda.io/en/latest/miniconda.html

echo === KWS-DPU Environment Setup ===
echo.

REM Check conda is available
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERROR] conda not found. Install Miniconda from:
    echo         https://docs.conda.io/en/latest/miniconda.html
    echo         Then re-run this script.
    pause
    exit /b 1
)

echo [1/4] Creating conda environment from environment.yml ...
conda env create -f environment.yml
if errorlevel 1 (
    echo [INFO] Environment may already exist. Updating instead ...
    conda env update -f environment.yml --prune
)

echo.
echo [2/4] Activating environment ...
call conda activate kws_dpu

echo.
echo [3/4] Verifying key packages ...
python -c "import torch; print('  torch       :', torch.__version__)"
python -c "import torchaudio; print('  torchaudio  :', torchaudio.__version__)"
python -c "import librosa; print('  librosa     :', librosa.__version__)"
python -c "import onnxruntime; print('  onnxruntime :', onnxruntime.__version__)"
python -c "import numpy; print('  numpy       :', numpy.__version__)"

echo.
echo [4/4] Saving library versions ...
python -c "
import sys
sys.path.insert(0, '.')
from pipeline.utils import save_versions, print_versions
from pathlib import Path
print_versions()
save_versions(Path('results/raw/versions_init.txt'))
"

echo.
echo [OK] Setup complete. Activate the environment with:
echo      conda activate kws_dpu
pause

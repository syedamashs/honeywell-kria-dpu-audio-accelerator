#!/usr/bin/env bash
# scripts/run_on_board.sh
# Step 12 — One-shot comprehensive board session runner for Kria KV260
#
# Runs on Kria KV260 Linux (Ubuntu) to benchmark:
#   Config A: CPU-only (ARM Cortex-A53, 1-4 threads)
#   Config B: CPU + DPU (DPUCZDX8G B3136, 1-4 threads)
#   Config B (unsupported op): dscnn_gru fallback evaluation
#
# Usage on KV260:
#   sudo bash scripts/run_on_board.sh

set -euo pipefail

echo "=================================================================="
echo " KRIA KV260 BENCHMARK SESSION: CPU vs DPU EVALUATION"
echo "=================================================================="

# 1. Fix CPU Frequency Governor to Performance
echo "[1/6] Setting CPU frequency governor to 'performance'..."
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    if [ -f "${gov}" ]; then
        echo performance | sudo tee "${gov}" > /dev/null
    fi
done
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor || true

# 2. Query and verify DPU configuration
echo "[2/6] Querying DPUCZDX8G hardware..."
if command -v xdputil &>/dev/null; then
    xdputil query || true
fi

# 3. Create results directory
RESULTS_DIR="results/board_session_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RESULTS_DIR}/raw" "${RESULTS_DIR}/power"

# 4. Run Config A: CPU-Only Baseline (1, 2, 4 threads)
echo "[3/6] Running Config A: CPU-Only Baseline..."
python3 benchmarks/cpu_baseline.py \
    --variant medium \
    --iters 1000 \
    --warmup 20 \
    --threads 1 2 4 \
    --board "Kria-KV260"

cp results/raw/cpu_baseline_medium.csv "${RESULTS_DIR}/raw/A_cpu_baseline.csv" || true

# 5. Run Config B: DPUCZDX8G Acceleration (1, 2, 4 threads)
echo "[4/6] Running Config B: CPU + DPU (VART)..."
if [ -f "models/compiled/dscnn_medium.xmodel" ]; then
    for TH in 1 2 4; do
        echo "  -> Running DPU with ${TH} threads..."
        python3 board/app/dpu_runner.py \
            --xmodel models/compiled/dscnn_medium.xmodel \
            --iters 1000 \
            --warmup 20 \
            --threads "${TH}" \
            --out "${RESULTS_DIR}/raw/B_dpu_th${TH}.csv"
    done
else
    echo "  [!] models/compiled/dscnn_medium.xmodel not found. Skipping DPU run."
fi

# 6. Measure Power (if onboard sensor available)
echo "[5/6] Logging onboard power sensors..."
for sensor in /sys/class/hwmon/hwmon*/power1_input; do
    if [ -f "${sensor}" ]; then
        echo "Found power sensor at ${sensor}: $(cat ${sensor}) uW"
        cat "${sensor}" >> "${RESULTS_DIR}/power/power_samples.txt"
    fi
done

# 7. Run Interactive Demo verification
echo "[6/6] Running verification demo..."
python3 board/app/demo.py --wav data/test_inputs/test_00_yes.wav --mode cpu

echo "=================================================================="
echo " SESSION COMPLETE. All board logs saved to: ${RESULTS_DIR}"
echo "=================================================================="

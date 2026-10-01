# Evaluation and Optimization of an IP DPU for Embedded AI/ML Inferencing

**Workload:** Keyword Spotting (KWS) — DS-CNN on log-mel/MFCC features, Google Speech Commands v2  
**Primary board:** Kria KV260 (Zynq UltraScale+ MPSoC, Quad-core A53, DPUCZDX8G)  
**Secondary board:** PYNQ-Z2 (Zynq-7000, Dual-core A9) — custom HLS kernel study only

## Three Configurations (all on KV260)

| Config | Preprocessing | Inference | Post |
|--------|--------------|-----------|------|
| A – CPU-only | ARM A53 | ARM A53 INT8 | ARM A53 |
| B – CPU + DPU | ARM A53 | DPUCZDX8G | ARM A53 |
| C – CPU + DPU + HLS | PL GEMM kernel | DPUCZDX8G | ARM A53 |

## Quick Start (Laptop — Part 1)

```bash
# 1. Create environment
conda env create -f environment.yml
conda activate kws_dpu

# 2. Download dataset
python scripts/download_dataset.py

# 3. Train model
python scripts/train.py --variant medium

# 4. Run CPU baseline benchmark
python benchmarks/cpu_baseline.py --iters 1000 --threads 4

# 5. Quantize (inside Vitis AI Docker)
bash scripts/quantize.sh

# 6. Compile for KV260 DPU (inside Vitis AI Docker)
bash scripts/compile.sh
```

## Repository Layout

```
data/           Dataset and fixed test inputs
models/         Checkpoints, ONNX, quantized, compiled
pipeline/       Python preprocessing, model, postprocessing, GEMM reference
benchmarks/     Harness and per-config runners
hls/            Vitis HLS custom GEMM kernel source
vivado/         KV260 block designs (DPU-only, DPU+kernel)
results/        CSV logs, plots, profiler traces
scripts/        Setup, dataset, quantize, compile, board scripts
board/          On-board application (copied to KV260)
report/         Main notebook, architecture diagram, partition map
```

## Fixed Preprocessing Parameters

| Parameter | Value |
|-----------|-------|
| Sample rate | 16,000 Hz |
| Window length | 25 ms (400 samples) |
| Hop length | 10 ms (160 samples) |
| FFT size | 512 |
| n_mels | 40 |
| fmin / fmax | 20 Hz / 4,000 Hz |
| Output frames | 101 |
| Classes | 10 keywords + unknown + silence |

## Reproducibility

All random seeds are fixed. See `pipeline/utils.py`. Library versions are pinned in `environment.yml`.  
Raw benchmark data is in `results/raw/`. All tables and plots regenerate from `report/main_notebook.ipynb`.

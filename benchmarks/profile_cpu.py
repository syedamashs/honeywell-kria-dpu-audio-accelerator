"""
benchmarks/profile_cpu.py
-------------------------
Step 8 — CPU hotspot profiling with cProfile and pstats.

Profiles the CPU pipeline end-to-end to identify exact runtime bottlenecks:
  - Framing & Hann windowing
  - FFT power spectrum
  - Mel filterbank GEMM
  - Log compression
  - ONNX runtime inference
  - Postprocessing

Outputs formatted profiling analysis to results/profiler/cpu_hotspots_{variant}.txt
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.cpu_baseline import CPUModelRunner, ensure_quantized_onnx, load_manifest
from pipeline.postprocessing import decode
from pipeline.preprocessing import extract_log_mel, load_wav

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
PROFILER_DIR = ROOT / "results" / "profiler"


def profile_pipeline(variant: str = "medium", iterations: int = 50, threads: int = 1):
    PROFILER_DIR.mkdir(parents=True, exist_ok=True)
    test_inputs = load_manifest()

    fp32_onnx = MODELS_DIR / "onnx" / f"dscnn_{variant}.onnx"
    int8_onnx = MODELS_DIR / "onnx" / f"dscnn_{variant}_int8.onnx"
    model_path = ensure_quantized_onnx(fp32_onnx, int8_onnx)

    runner = CPUModelRunner(model_path, threads=threads)

    def run_iterations():
        for i in range(iterations):
            item = test_inputs[i % len(test_inputs)]
            raw_audio = load_wav(item["path"])
            features = extract_log_mel(raw_audio, apply_pre_emphasis=True)
            logits = runner(features)
            label, conf, _ = decode(logits)

    print(f"[*] Profiling CPU pipeline ({variant}, {iterations} iterations, {threads} thread)...")
    profiler = cProfile.Profile()
    profiler.enable()
    run_iterations()
    profiler.disable()

    out_file = PROFILER_DIR / f"cpu_hotspots_{variant}.txt"
    with open(out_file, "w") as f:
        ps = pstats.Stats(profiler, stream=f)
        ps.strip_dirs()
        ps.sort_stats("cumulative")
        ps.print_stats(35)
        ps.sort_stats("time")
        ps.print_stats(35)

    print(f"[OK] Profiler summary written -> {out_file}")

    # Print top cumulative time directly to console
    print("\n--- TOP CPU FUNCTIONS BY CUMULATIVE TIME ---")
    ps_console = pstats.Stats(profiler)
    ps_console.strip_dirs()
    ps_console.sort_stats("cumulative")
    ps_console.print_stats(15)


def main():
    parser = argparse.ArgumentParser(description="Profile CPU Pipeline Hotspots")
    parser.add_argument("--variant", default="medium", choices=["small", "medium", "large", "gru"])
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    profile_pipeline(variant=args.variant, iterations=args.iters, threads=args.threads)


if __name__ == "__main__":
    main()

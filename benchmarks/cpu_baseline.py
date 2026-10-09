"""
benchmarks/cpu_baseline.py
--------------------------
Step 7 — CPU-only baseline (Config A) implementation and benchmark.

Features:
  - Quantizes FP32 ONNX model to INT8 via onnxruntime.quantization
  - Runs full end-to-end pipeline on CPU:
      WAV load -> Preprocessing (log-mel) -> ONNX INT8/FP32 -> Postprocessing
  - Sweeps threads (1, 2, 4) to measure multi-core scaling
  - Evaluates on 10 fixed evaluation test clips
  - Outputs per-stage and end-to-end timing to CSV and terminal summary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

os.environ["ORT_LOGGING_LEVEL"] = "3"
import onnxruntime as ort
try:
    ort.set_default_logger_severity(3)
except Exception:
    pass
try:
    from onnxruntime.quantization import QuantType, quantize_dynamic
except (ImportError, Exception):
    QuantType = None
    quantize_dynamic = None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.harness import BenchmarkRunner
from pipeline.postprocessing import decode
from pipeline.preprocessing import extract_log_mel, load_wav
from pipeline.utils import GLOBAL_SEED, set_seed

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "raw"


def ensure_quantized_onnx(fp32_path: Path, int8_path: Path) -> Path:
    """Quantize FP32 ONNX model to INT8 using dynamic quantization if not already present."""
    if int8_path.exists():
        return int8_path

    print(f"[*] Preparing and quantizing {fp32_path.name} to INT8 dynamic...")
    int8_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean stale value_info and ensure self-contained weights
    import onnx
    m = onnx.load(str(fp32_path), load_external_data=True)
    while len(m.graph.value_info) > 0:
        m.graph.value_info.pop()
    m = onnx.shape_inference.infer_shapes(m, check_type=True)
    clean_fp32 = fp32_path.with_name(f"{fp32_path.stem}_clean.onnx")
    onnx.save_model(m, str(clean_fp32), save_as_external_data=False)

    quantize_dynamic(
        model_input=str(clean_fp32),
        model_output=str(int8_path),
        weight_type=QuantType.QInt8,
    )
    if clean_fp32.exists():
        clean_fp32.unlink(missing_ok=True)

    print(f"[OK] Quantized model saved -> {int8_path}")
    return int8_path


def load_manifest() -> List[Dict[str, Any]]:
    manifest_path = DATA_DIR / "test_inputs" / "test_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Test manifest not found at {manifest_path}. "
            "Run scripts/generate_synthetic_test_data.py first."
        )
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    return list(manifest.values())


class CPUModelRunner:
    def __init__(self, model_path: Path, threads: int = 1):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.log_verbosity_level = 0
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, features: np.ndarray) -> np.ndarray:
        # features: (40, 98) -> reshape to (1, 1, 40, 98)
        x = features[np.newaxis, np.newaxis, :, :].astype(np.float32)
        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        return logits[0]  # return shape (12,)


def main():
    parser = argparse.ArgumentParser(description="CPU-only Baseline Benchmark (Config A)")
    parser.add_argument("--variant", default="medium", choices=["small", "medium", "large", "gru"])
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="Threads to sweep")
    parser.add_argument("--precision", choices=["both", "fp32", "int8"], default="both")
    parser.add_argument("--board", default="Host-CPU", help="Board identifier (e.g. KV260 or Host-CPU)")
    args = parser.parse_args()

    set_seed(GLOBAL_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_inputs = load_manifest()
    print(f"[*] Loaded {len(test_inputs)} fixed test inputs from manifest.")

    fp32_onnx = MODELS_DIR / "onnx" / f"dscnn_{args.variant}.onnx"
    int8_onnx = MODELS_DIR / "onnx" / f"dscnn_{args.variant}_int8.onnx"

    if not fp32_onnx.exists():
        raise FileNotFoundError(f"Model {fp32_onnx} not found. Run scripts/train.py first.")

    precisions = []
    if args.precision in ["fp32", "both"]:
        precisions.append(("FP32", fp32_onnx))
    if args.precision in ["int8", "both"]:
        ensure_quantized_onnx(fp32_onnx, int8_onnx)
        precisions.append(("INT8", int8_onnx))

    csv_path = RESULTS_DIR / f"cpu_baseline_{args.variant}.csv"

    # Preprocessing & postprocessing functions
    load_fn = load_wav
    preproc_fn = lambda wav: extract_log_mel(wav, apply_pre_emphasis=True)
    postproc_fn = decode

    all_summaries = []

    for prec_name, model_file in precisions:
        for th in args.threads:
            config_label = f"Config-A-CPU-{prec_name}-{args.variant}-th{th}"
            runner = CPUModelRunner(model_file, threads=th)
            bench = BenchmarkRunner(
                board=args.board,
                config_name=config_label,
                threads=th,
                batch_size=1,
            )

            _, summary = bench.run_benchmark(
                test_inputs=test_inputs,
                load_fn=load_fn,
                preproc_fn=preproc_fn,
                infer_fn=runner,
                postproc_fn=postproc_fn,
                num_iterations=args.iters,
                num_warmup=args.warmup,
                csv_output_path=csv_path,
            )
            all_summaries.append(summary)

    # Print summary comparison table
    print("\n" + "=" * 90)
    print(f"CPU BASELINE COMPARISON SUMMARY ({args.variant.upper()})")
    print("=" * 90)
    print(f"{'Configuration':<35} {'Threads':>8} {'Infer (ms)':>12} {'Total (ms)':>12} {'FPS':>10}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s.config_name:<35} {s.threads:>8} {s.infer_median_ms:>12.2f} {s.total_median_ms:>12.2f} {s.throughput_fps:>10.2f}")
    print("=" * 90)
    print(f"[OK] Raw benchmark logs written -> {csv_path}\n")


if __name__ == "__main__":
    main()

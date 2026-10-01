"""
board/app/demo.py
-----------------
Step 12 — Interactive Demo Script for Keyword Spotting (KWS).

Demonstrates the audio KWS pipeline on a chosen audio clip (or microphone input):
  - Prints recognized keyword and confidence score
  - Displays formatted latency breakdown across all 4 stages:
      1. WAV Audio Load
      2. Mel Preprocessing (FFT + GEMM + Log)
      3. Model Inference (DPU / CPU)
      4. Softmax Postprocessing & Decoding
  - Works on KV260 board (with DPU) and host laptop (with CPU fallback)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.postprocessing import decode
from pipeline.preprocessing import extract_log_mel, load_wav


def run_demo(
    wav_path: Path,
    mode: str = "cpu",
    model_path: Path | None = None,
):
    print("=" * 70)
    print(" EMBEDDED AUDIO AI/ML KEYWORD SPOTTING DEMO")
    print(f" Target Mode: {mode.upper()} | Input: {wav_path.name}")
    print("=" * 70)

    # 1. Load Audio
    t0 = time.perf_counter_ns()
    wav = load_wav(wav_path)
    t1 = time.perf_counter_ns()

    # 2. Preprocess (Framing, FFT, Mel Filterbank GEMM, Log)
    t2 = time.perf_counter_ns()
    features = extract_log_mel(wav, apply_pre_emphasis=True)
    t3 = time.perf_counter_ns()

    # 3. Model Inference
    t4 = time.perf_counter_ns()
    if mode == "dpu":
        from board.app.dpu_runner import VARTDPURunner
        runner = VARTDPURunner(model_path or "models/compiled/dscnn_medium.xmodel")
        logits, _ = runner.infer(features)
    else:
        from benchmarks.cpu_baseline import CPUModelRunner
        default_model = ROOT / "models" / "onnx" / "dscnn_medium.onnx"
        runner = CPUModelRunner(model_path or default_model)
        logits = runner(features)
    t5 = time.perf_counter_ns()

    # 4. Postprocessing (Softmax, Argmax, Decode)
    t6 = time.perf_counter_ns()
    label, confidence, class_idx = decode(logits)
    t7 = time.perf_counter_ns()

    load_ms = (t1 - t0) / 1e6
    preproc_ms = (t3 - t2) / 1e6
    infer_ms = (t5 - t4) / 1e6
    post_ms = (t7 - t6) / 1e6
    total_ms = load_ms + preproc_ms + infer_ms + post_ms

    # Visual Output
    print(f"\n >>> DETECTED KEYWORD: [{label.upper()}] <<<")
    print(f" Confidence Score : {confidence * 100:.2f}%")
    print(f" Class Index      : {class_idx}\n")

    print(" Stage Latency Breakdown:")
    print(" -------------------------------------------------------------")
    print(f"   [1] Audio Load (WAV IO)     : {load_ms:6.2f} ms ({load_ms/total_ms*100:4.1f}%)")
    print(f"   [2] Mel Preprocessing (GEMM): {preproc_ms:6.2f} ms ({preproc_ms/total_ms*100:4.1f}%)")
    print(f"   [3] Model Inference ({mode.upper()}): {infer_ms:6.2f} ms ({infer_ms/total_ms*100:4.1f}%)")
    print(f"   [4] Softmax Postprocessing  : {post_ms:6.2f} ms ({post_ms/total_ms*100:4.1f}%)")
    print(" -------------------------------------------------------------")
    print(f"   TOTAL END-TO-END PIPELINE   : {total_ms:6.2f} ms (Throughput: {1000.0/total_ms:5.1f} FPS)")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="KWS Audio Demo")
    parser.add_argument("--wav", default="data/test_inputs/test_00_yes.wav", help="WAV file path")
    parser.add_argument("--mode", default="cpu", choices=["cpu", "dpu"], help="Execution mode")
    parser.add_argument("--model", type=str, default=None, help="Model file path")
    args = parser.parse_args()

    run_demo(
        wav_path=ROOT / args.wav,
        mode=args.mode,
        model_path=Path(args.model) if args.model else None,
    )


if __name__ == "__main__":
    main()

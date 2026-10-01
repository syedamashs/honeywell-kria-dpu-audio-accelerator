"""
board/app/demo.py
-----------------
Step 12 — Dual-Mode Interactive Demo for Keyword Spotting (KWS) on Kria KV260.

Supports TWO INPUT MODES:
  1. [ Passive Data Mode ]:
     Uses stored voice audio from Google Speech Commands dataset or WAV files.
  2. [ Real-Time Voice Mode ]:
     Captures live speech directly from the microphone.

CRITICAL ARCHITECTURE PROPERTY:
Both modes output the EXACT SAME standardized 16 kHz audio representation (16,000 samples)
and traverse the EXACT SAME preprocessing, feature representation [1, 1, 40, 98], and
hardware inference pipeline (Config A: CPU or Config B: DPUCZDX8G).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.audio_stream import get_audio_input
from pipeline.postprocessing import decode
from pipeline.preprocessing import extract_log_mel


def run_demo(
    input_mode: str = "passive",
    wav_path: Path | None = None,
    engine_mode: str = "cpu",
    model_path: Path | None = None,
):
    print("\n" + "=" * 75)
    print("      AMD KRIA KV260 -- EMBEDDED AUDIO AI/ML KEYWORD SPOTTING DEMO")
    print("=" * 75)
    print(f" Selected Input Mode : [{'PASSIVE DATA' if input_mode == 'passive' else 'REAL-TIME VOICE'}]")
    print(f" Hardware Engine     : {engine_mode.upper()} ({'ARM Cortex-A53' if engine_mode == 'cpu' else 'DPUCZDX8G IP Core'})")
    print("=" * 75)

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Audio Acquisition (Passive Dataset vs Real-Time Microphone)
    # ─────────────────────────────────────────────────────────────────────────
    t0 = time.perf_counter_ns()
    if input_mode == "passive":
        default_wav = ROOT / "data" / "test_inputs" / "test_00_yes.wav"
        target_file = wav_path or default_wav
        audio, src_desc = get_audio_input("passive", wav_path=target_file)
    else:
        audio, src_desc = get_audio_input("realtime", capture_duration_s=1.0)
    t1 = time.perf_counter_ns()

    print(f"[*] Input Source: {src_desc}")
    print(f"[*] Raw Audio Shape: {audio.shape} @ 16 kHz ({len(audio)/16000:.2f}s)")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: Preprocessing (Identical for BOTH modes)
    # ─────────────────────────────────────────────────────────────────────────
    t2 = time.perf_counter_ns()
    features = extract_log_mel(audio, apply_pre_emphasis=True)
    t3 = time.perf_counter_ns()

    print(f"[*] Preprocessed Feature Tensor: shape {features.shape} (40 mel bins x 98 frames)")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 3: Model Inference (Identical for BOTH modes)
    # ─────────────────────────────────────────────────────────────────────────
    t4 = time.perf_counter_ns()
    if engine_mode == "dpu":
        from board.app.dpu_runner import VARTDPURunner
        default_xmodel = ROOT / "models" / "compiled" / "dscnn_medium.xmodel"
        runner = VARTDPURunner(model_path or default_xmodel)
        logits, _ = runner.infer(features)
    else:
        from benchmarks.cpu_baseline import CPUModelRunner
        default_onnx = ROOT / "models" / "onnx" / "dscnn_medium.onnx"
        runner = CPUModelRunner(model_path or default_onnx)
        logits = runner(features)
    t5 = time.perf_counter_ns()

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 4: Softmax Postprocessing & Label Decoding
    # ─────────────────────────────────────────────────────────────────────────
    t6 = time.perf_counter_ns()
    label, confidence, class_idx = decode(logits)
    t7 = time.perf_counter_ns()

    # Compute Latency Metrics
    acq_ms = (t1 - t0) / 1e6
    preproc_ms = (t3 - t2) / 1e6
    infer_ms = (t5 - t4) / 1e6
    post_ms = (t7 - t6) / 1e6
    pipeline_ms = preproc_ms + infer_ms + post_ms
    total_ms = acq_ms + pipeline_ms

    # Visual Output
    print("\n" + "-" * 75)
    print(f" >>> DETECTED KEYWORD: [ {label.upper()} ] <<<")
    print(f" Confidence Score    : {confidence * 100:.2f}%")
    print(f" Class Index         : {class_idx}")
    print("-" * 75)

    print("\n Execution Latency Breakdown:")
    print(" -------------------------------------------------------------------------")
    print(f"   [1] Audio Ingestion ({'WAV IO' if input_mode == 'passive' else 'Live Mic'}) : {acq_ms:6.2f} ms")
    print(f"   [2] Mel Preprocessing (GEMM)     : {preproc_ms:6.2f} ms ({preproc_ms/pipeline_ms*100:4.1f}% of pipeline)")
    print(f"   [3] Model Inference ({engine_mode.upper():<11}) : {infer_ms:6.2f} ms ({infer_ms/pipeline_ms*100:4.1f}% of pipeline)")
    print(f"   [4] Softmax Postprocessing       : {post_ms:6.2f} ms ({post_ms/pipeline_ms*100:4.1f}% of pipeline)")
    print(" -------------------------------------------------------------------------")
    print(f"   CORE INFERENCE PIPELINE LATENCY  : {pipeline_ms:6.2f} ms (Throughput: {1000.0/pipeline_ms:5.1f} FPS)")
    print("=" * 75 + "\n")


def interactive_menu():
    """Interactive CLI menu when run without flags."""
    print("=" * 70)
    print(" SELECT AUDIO INPUT MODE:")
    print("   [1] Passive Data Mode    (Stored audio dataset file / WAV)")
    print("   [2] Real-Time Voice Mode (Live microphone speech)")
    print("=" * 70)
    try:
        choice = input("Enter choice [1 or 2, default=1]: ").strip()
    except EOFError:
        choice = "1"

    if choice == "2":
        return "realtime"
    return "passive"


def main():
    parser = argparse.ArgumentParser(description="KWS Audio Demo (Passive Data vs Real-Time Voice)")
    parser.add_argument(
        "--input-mode",
        choices=["passive", "realtime", "interactive"],
        default="passive",
        help="Input mode: 'passive' (dataset WAV) or 'realtime' (microphone) or 'interactive' prompt",
    )
    parser.add_argument("--wav", default=None, help="Path to WAV audio file (for passive mode)")
    parser.add_argument("--engine", default="cpu", choices=["cpu", "dpu"], help="Inference hardware engine")
    parser.add_argument("--model", default=None, help="Path to model file (.onnx or .xmodel)")
    args = parser.parse_args()

    mode = args.input_mode
    if mode == "interactive":
        mode = interactive_menu()

    wav_p = Path(args.wav) if args.wav else None
    run_demo(
        input_mode=mode,
        wav_path=wav_p,
        engine_mode=args.engine,
        model_path=Path(args.model) if args.model else None,
    )


if __name__ == "__main__":
    main()

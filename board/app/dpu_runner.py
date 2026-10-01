"""
board/app/dpu_runner.py
-----------------------
Step 12 — Board-ready VART DPU application for Kria KV260 (Config B).

Features:
  - Loads compiled .xmodel via XIR
  - Creates VART Runner for DPUCZDX8G subgraph
  - Handles NHWC tensor layout transposition and int8 fixed-point scaling
  - Executes asynchronous DPU inference jobs with precise timestamping
  - Multi-threaded execution harness for throughput sweeps
  - Exports per-stage latency logs to CSV
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.postprocessing import decode
from pipeline.preprocessing import extract_log_mel, load_wav


class VARTDPURunner:
    """
    Manages VART Runner lifecycle on Kria KV260.
    """

    def __init__(self, xmodel_path: str | Path):
        self.xmodel_path = Path(xmodel_path)
        if not self.xmodel_path.exists():
            raise FileNotFoundError(f"Compiled model not found: {self.xmodel_path}")

        try:
            import vart
            import xir
        except ImportError:
            raise ImportError(
                "VART / XIR libraries not found. Ensure this script is run on the KV260 board "
                "with Vitis AI runtime installed, or inside the Vitis AI Docker container."
            )

        # Deserialize graph and locate DPU subgraph
        self.graph = xir.Graph.deserialize(str(self.xmodel_path))
        root_subgraph = self.graph.get_root_subgraph()
        dpu_subgraphs = [
            sg for sg in root_subgraph.get_children()
            if sg.has_attr("device") and sg.get_attr("device") == "DPU"
        ]

        if not dpu_subgraphs:
            raise RuntimeError(f"No DPU subgraph found in {self.xmodel_path}")

        self.dpu_subgraph = dpu_subgraphs[0]
        self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")

        # Tensor shapes (DPU uses NHWC)
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        self.in_shape = tuple(self.input_tensors[0].dims)   # e.g., (1, 98, 40, 1) or (1, 40, 98, 1)
        self.out_shape = tuple(self.output_tensors[0].dims) # e.g., (1, 12)

        # Quantization fix-point scale (if available)
        self.fix_pos = self.input_tensors[0].get_attr("fix_point") if self.input_tensors[0].has_attr("fix_point") else 0
        self.scale = 2.0 ** self.fix_pos

    def infer(self, features: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Execute single DPU inference.
        Args:
            features: (40, 98) float32 log-mel features
        Returns:
            (logits, dpu_execution_time_ns)
        """
        # Allocate buffers
        in_buf = [np.zeros(self.in_shape, dtype=np.int8)]
        out_buf = [np.zeros(self.out_shape, dtype=np.int8)]

        # Prepare input: scale and convert to INT8
        # PyTorch NCHW (1, 1, 40, 98) -> DPU NHWC (1, 40, 98, 1)
        feat_scaled = np.clip(features * self.scale, -128, 127).astype(np.int8)
        if len(self.in_shape) == 4:
            if self.in_shape[1] == 40 and self.in_shape[2] == 98:
                in_buf[0][0, :, :, 0] = feat_scaled
            elif self.in_shape[1] == 98 and self.in_shape[2] == 40:
                in_buf[0][0, :, :, 0] = feat_scaled.T

        # Timed execution
        t0 = time.perf_counter_ns()
        job_id = self.runner.execute_async(in_buf, out_buf)
        self.runner.wait(job_id)
        t1 = time.perf_counter_ns()

        dpu_time_ns = t1 - t0
        logits = out_buf[0][0].astype(np.float32)
        return logits, dpu_time_ns


def run_board_benchmark(
    xmodel_path: Path,
    manifest_path: Path,
    iterations: int = 1000,
    warmup: int = 20,
    threads: int = 1,
    out_csv: Path = ROOT / "results" / "raw" / "board_dpu_results.csv",
):
    with open(manifest_path, "r") as f:
        manifest = list(json.load(f).values())

    dpu_runners = [VARTDPURunner(xmodel_path) for _ in range(threads)]
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fh = open(out_csv, "w", newline="")
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "board", "config", "input_id", "iteration", "thread_id",
            "load_ns", "preproc_ns", "infer_ns", "post_ns", "total_ns",
            "prediction", "confidence", "correct",
        ],
    )
    writer.writeheader()
    lock = threading.Lock()

    print(f"[*] Starting KV260 DPU Benchmark: {threads} threads, {iterations} iters/thread...")

    def worker(th_id: int):
        runner = dpu_runners[th_id]
        n_items = len(manifest)

        for i in range(warmup + iterations):
            is_warmup = i < warmup
            item = manifest[i % n_items]
            wav_path = item["path"]
            gt = item["label"]

            t0 = time.perf_counter_ns()
            wav = load_wav(wav_path)
            t1 = time.perf_counter_ns()
            feats = extract_log_mel(wav, apply_pre_emphasis=True)
            t2 = time.perf_counter_ns()
            logits, dpu_ns = runner.infer(feats)
            t3 = time.perf_counter_ns()
            label, conf, _ = decode(logits)
            t4 = time.perf_counter_ns()

            if not is_warmup:
                with lock:
                    writer.writerow({
                        "board": "Kria-KV260",
                        "config": "Config-B-DPUCZDX8G",
                        "input_id": item["filename"],
                        "iteration": i - warmup,
                        "thread_id": th_id,
                        "load_ns": t1 - t0,
                        "preproc_ns": t2 - t1,
                        "infer_ns": dpu_ns,
                        "post_ns": t4 - t3,
                        "total_ns": t4 - t0,
                        "prediction": label,
                        "confidence": f"{conf:.4f}",
                        "correct": (label == gt),
                    })

    threads_list = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
    t_start = time.time()
    for th in threads_list:
        th.start()
    for th in threads_list:
        th.join()
    t_end = time.time()
    fh.close()

    total_infs = threads * iterations
    fps = total_infs / (t_end - t_start)
    print(f"[OK] KV260 DPU Benchmark Complete.")
    print(f"     Throughput: {fps:.2f} inferences/second")
    print(f"     Logs written -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Kria KV260 VART DPU Runner")
    parser.add_argument("--xmodel", default="models/compiled/dscnn_medium.xmodel")
    parser.add_argument("--manifest", default="data/test_inputs/test_manifest.json")
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", default="results/raw/board_dpu_results.csv")
    args = parser.parse_args()

    run_board_benchmark(
        xmodel_path=ROOT / args.xmodel,
        manifest_path=ROOT / args.manifest,
        iterations=args.iters,
        warmup=args.warmup,
        threads=args.threads,
        out_csv=ROOT / args.out,
    )


if __name__ == "__main__":
    main()

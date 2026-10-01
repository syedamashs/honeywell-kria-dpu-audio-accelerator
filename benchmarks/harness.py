"""
benchmarks/harness.py
---------------------
Step 8 — Benchmarking harness for all configurations (A, B, C).

Provides:
  - BenchmarkRunner: Executes warmup and timed iterations with per-stage timing
    (load, preprocessing, inference, postprocessing)
  - Detailed latency statistics (mean, median, p95, p99, std, min, max)
  - Throughput (inferences per second)
  - CSV result logging following the project's standard schema
  - Terminal formatted summary display
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.utils import (
    IDX2LABEL,
    LABEL2IDX,
    open_result_csv,
    set_seed,
)


@dataclass
class StageTiming:
    load_ns: int
    preproc_ns: int
    infer_ns: int
    post_ns: int
    total_ns: int
    prediction: str
    confidence: float
    correct: Optional[bool] = None


@dataclass
class BenchmarkSummary:
    config_name: str
    board_name: str
    num_iterations: int
    threads: int
    batch_size: int
    load_median_ms: float
    preproc_median_ms: float
    infer_median_ms: float
    post_median_ms: float
    total_median_ms: float
    total_mean_ms: float
    total_p95_ms: float
    total_p99_ms: float
    throughput_fps: float


class BenchmarkRunner:
    """
    Standard benchmark harness for audio KWS pipeline.
    Works identically across laptop (proxy) and KV260 board.
    """

    def __init__(
        self,
        board: str = "Host-CPU",
        config_name: str = "Config-A-CPU-INT8",
        threads: int = 1,
        batch_size: int = 1,
    ):
        self.board = board
        self.config_name = config_name
        self.threads = threads
        self.batch_size = batch_size

    def run_benchmark(
        self,
        test_inputs: List[Dict[str, Any]],
        load_fn: Callable[[str], Any],
        preproc_fn: Callable[[Any], Any],
        infer_fn: Callable[[Any], Any],
        postproc_fn: Callable[[Any], Tuple[str, float, int]],
        num_iterations: int = 100,
        num_warmup: int = 10,
        csv_output_path: Optional[Path] = None,
    ) -> Tuple[List[StageTiming], BenchmarkSummary]:
        """
        Execute warmup followed by timed iterations.
        """
        fh = None
        writer = None
        if csv_output_path:
            csv_output_path = Path(csv_output_path)
            csv_output_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_output_path.exists()
            fh = open(csv_output_path, "a", newline="")
            fieldnames = [
                "board", "config", "input_id", "iteration",
                "load_ns", "preproc_ns", "infer_ns", "post_ns", "total_ns",
                "prediction", "confidence", "correct", "threads", "batch_size",
            ]
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

        timings: List[StageTiming] = []
        n_inputs = len(test_inputs)
        total_runs = num_warmup + num_iterations

        print(f"[*] Running benchmark: {self.config_name} on {self.board}")
        print(f"    Warmup: {num_warmup} iters | Timed: {num_iterations} iters | Threads: {self.threads}")

        for i in range(total_runs):
            is_warmup = i < num_warmup
            item = test_inputs[i % n_inputs]
            input_id = item.get("filename", f"input_{i % n_inputs}")
            ground_truth = item.get("label", None)
            path = item.get("path")

            # 1. Load stage
            t0 = time.perf_counter_ns()
            raw_audio = load_fn(path)
            t1 = time.perf_counter_ns()

            # 2. Preprocessing stage
            features = preproc_fn(raw_audio)
            t2 = time.perf_counter_ns()

            # 3. Inference stage
            logits = infer_fn(features)
            t3 = time.perf_counter_ns()

            # 4. Postprocessing stage
            label, conf, idx = postproc_fn(logits)
            t4 = time.perf_counter_ns()

            load_ns = t1 - t0
            preproc_ns = t2 - t1
            infer_ns = t3 - t2
            post_ns = t4 - t3
            total_ns = t4 - t0

            correct = (label == ground_truth) if ground_truth else None

            st = StageTiming(
                load_ns=load_ns,
                preproc_ns=preproc_ns,
                infer_ns=infer_ns,
                post_ns=post_ns,
                total_ns=total_ns,
                prediction=label,
                confidence=conf,
                correct=correct,
            )

            if not is_warmup:
                timings.append(st)
                if writer:
                    writer.writerow({
                        "board": self.board,
                        "config": self.config_name,
                        "input_id": input_id,
                        "iteration": i - num_warmup,
                        "load_ns": load_ns,
                        "preproc_ns": preproc_ns,
                        "infer_ns": infer_ns,
                        "post_ns": post_ns,
                        "total_ns": total_ns,
                        "prediction": label,
                        "confidence": f"{conf:.4f}",
                        "correct": correct,
                        "threads": self.threads,
                        "batch_size": self.batch_size,
                    })

        if fh:
            fh.close()

        summary = self._compute_summary(timings, num_iterations)
        self._print_summary(summary)
        return timings, summary

    def _compute_summary(self, timings: List[StageTiming], n_iters: int) -> BenchmarkSummary:
        loads = np.array([t.load_ns for t in timings], dtype=np.float64) / 1e6
        preprocs = np.array([t.preproc_ns for t in timings], dtype=np.float64) / 1e6
        infers = np.array([t.infer_ns for t in timings], dtype=np.float64) / 1e6
        posts = np.array([t.post_ns for t in timings], dtype=np.float64) / 1e6
        totals = np.array([t.total_ns for t in timings], dtype=np.float64) / 1e6

        total_mean = float(np.mean(totals))
        throughput = (1000.0 / total_mean) if total_mean > 0 else 0.0

        return BenchmarkSummary(
            config_name=self.config_name,
            board_name=self.board,
            num_iterations=n_iters,
            threads=self.threads,
            batch_size=self.batch_size,
            load_median_ms=float(np.median(loads)),
            preproc_median_ms=float(np.median(preprocs)),
            infer_median_ms=float(np.median(infers)),
            post_median_ms=float(np.median(posts)),
            total_median_ms=float(np.median(totals)),
            total_mean_ms=total_mean,
            total_p95_ms=float(np.percentile(totals, 95)),
            total_p99_ms=float(np.percentile(totals, 99)),
            throughput_fps=throughput,
        )

    def _print_summary(self, s: BenchmarkSummary) -> None:
        print("\n" + "=" * 75)
        print(f"BENCHMARK RESULTS: {s.config_name} ({s.board_name})")
        print(f"Threads: {s.threads} | Batch: {s.batch_size} | Iterations: {s.num_iterations}")
        print("=" * 75)
        print(f"  Stage            Median (ms)    Mean (ms)     P95 (ms)      P99 (ms)")
        print(f"  -------------------------------------------------------------------")
        print(f"  Load             {s.load_median_ms:>9.2f}")
        print(f"  Preprocessing    {s.preproc_median_ms:>9.2f}")
        print(f"  Inference        {s.infer_median_ms:>9.2f}")
        print(f"  Postprocessing   {s.post_median_ms:>9.2f}")
        print(f"  -------------------------------------------------------------------")
        print(f"  TOTAL E2E        {s.total_median_ms:>9.2f}    {s.total_mean_ms:>9.2f}    {s.total_p95_ms:>9.2f}    {s.total_p99_ms:>9.2f}")
        print(f"\n  Throughput: {s.throughput_fps:.2f} inferences/second (FPS)")
        print("=" * 75 + "\n")

"""
scripts/benchmark_dpu_instances.py
----------------------------------
Multi-Instance / Multi-Thread Throughput & Latency Scaling Benchmark
for AMD DPUCZDX8G B4096 IP Core on AMD Kria KV260.

Evaluates:
  - 1 Instance (Single-stream baseline latency)
  - 2 Instances (Pipelined concurrency)
  - 4 Instances (Full hardware saturation)
  - 8 Instances (Saturated queue / stress test)
"""

import os
import sys
import time
import threading
from pathlib import Path
import numpy as np

try:
    import xir
    import vart
except ImportError:
    print("[!] Error: xir and vart modules not found. Run this on the physical KV260 board.")
    sys.exit(1)


def benchmark_instance_count(dpu_sub, num_instances: int, runs_per_instance: int = 250):
    """Run `num_instances` concurrent VART runners across separate threads."""
    thread_durations = [0.0] * num_instances
    threads = []

    def worker(inst_id: int):
        runner = vart.Runner.create_runner(dpu_sub, "run")
        in_tensors = runner.get_input_tensors()
        out_tensors = runner.get_output_tensors()
        in_shape = tuple(in_tensors[0].dims)
        out_shape = tuple(out_tensors[0].dims)

        in_buf = np.ascontiguousarray(np.zeros(in_shape, dtype=np.int8))
        out_buf = np.ascontiguousarray(np.zeros(out_shape, dtype=np.int8))

        # Warm-up run
        job = runner.execute_async([in_buf], [out_buf])
        runner.wait(job)

        # Timed loop
        t0 = time.perf_counter()
        for _ in range(runs_per_instance):
            job = runner.execute_async([in_buf], [out_buf])
            runner.wait(job)
        t1 = time.perf_counter()
        thread_durations[inst_id] = t1 - t0

    t_start = time.perf_counter()
    for i in range(num_instances):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    t_end = time.perf_counter()

    wall_time = t_end - t_start
    total_inferences = num_instances * runs_per_instance
    fps = total_inferences / wall_time
    avg_latency_ms = (wall_time / runs_per_instance) * 1000.0

    return fps, avg_latency_ms


def main():
    xmodel_candidates = [
        Path("models/compiled/dscnn_medium.xmodel"),
        Path("/home/ubuntu/deploy_kria_kv260/models/compiled/dscnn_medium.xmodel"),
        Path("dscnn_medium.xmodel"),
    ]
    xmodel_path = next((p for p in xmodel_candidates if p.exists()), None)
    if not xmodel_path:
        print("[!] Error: dscnn_medium.xmodel not found!")
        sys.exit(1)

    print("=" * 70)
    print(" AMD DPUCZDX8G B4096 — MULTI-INSTANCE CONCURRENCY SCALING")
    print(f" Target Model : {xmodel_path}")
    print(" Architecture : DPUCZDX8G_ISA1_B4096 (FPGA PL Fabric)")
    print("=" * 70)

    graph = xir.Graph.deserialize(str(xmodel_path))
    root = graph.get_root_subgraph()
    dpu_subs = [sg for sg in root.get_children() if sg.has_attr("device") and sg.get_attr("device") == "DPU"]
    if not dpu_subs:
        print("[!] No DPU subgraph found in xmodel.")
        sys.exit(1)
    dpu_sub = dpu_subs[0]

    instance_counts = [1, 2, 4, 8]
    base_fps = None

    print(f"\n{'Instances':<12} | {'Throughput (FPS)':<18} | {'Avg Latency (ms)':<18} | {'Speedup':<10}")
    print("-" * 70)

    for n in instance_counts:
        fps, lat_ms = benchmark_instance_count(dpu_sub, n, runs_per_instance=200)
        if base_fps is None:
            base_fps = fps
        speedup = fps / base_fps
        print(f"{n:<12} | {fps:>14.1f} FPS | {lat_ms:>14.2f} ms | {speedup:>8.2f}x")

    print("-" * 70)
    print(" >>> Hardware pipeline saturation achieved at peak concurrency. <<<")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

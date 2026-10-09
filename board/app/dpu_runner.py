"""
board/app/dpu_runner.py
-----------------------
Step 12 — Board-ready VART DPU application for Kria KV260 (Config B).

Features:
  - Loads compiled .xmodel via XIR
  - Creates VART Runner for DPUCZDX8G B4096 subgraph
  - Handles NHWC tensor layout transposition and int8 fixed-point scaling
  - Uses strictly C-contiguous buffers for zero-copy DMA execution
  - Singleton runner caching for ultra-low latency inference
  - Hardware interrupt (GIC IRQ) telemetry tracking
"""

from __future__ import annotations

import argparse
import csv
import glob
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


def get_dpu_irq_count() -> int:
    """Reads the current hardware interrupt count for zocl_cu from /proc/interrupts."""
    try:
        with open("/proc/interrupts", "r") as f:
            for line in f:
                if "zocl" in line.lower():
                    parts = line.split()
                    return int(parts[1])
    except Exception:
        pass
    return 0


class VARTDPURunner:
    """
    Manages VART Runner lifecycle on Kria KV260.
    """
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, xmodel_path: str | Path | None = None) -> VARTDPURunner:
        """Returns or creates the cached singleton instance of VARTDPURunner."""
        with cls._lock:
            if cls._instance is None:
                if xmodel_path is None:
                    xmodel_path = ROOT / "models" / "compiled" / "dscnn_medium.xmodel"
                cls._instance = cls(xmodel_path)
            return cls._instance

    def __init__(self, xmodel_path: str | Path):
        self.xmodel_path = Path(xmodel_path)
        if not self.xmodel_path.exists():
            # Check fallback locations
            fallbacks = [
                ROOT / "models" / "compiled" / "dscnn_medium.xmodel",
                Path("models/compiled/dscnn_medium.xmodel"),
                Path("dscnn_medium.xmodel"),
                ROOT / "models" / "nndct_xmodel" / "RecoveredDSCNN_int.xmodel",
            ]
            for fb in fallbacks:
                if fb.exists():
                    self.xmodel_path = fb
                    break
            else:
                raise FileNotFoundError(f"Compiled model not found: {self.xmodel_path}")

        # 1. Clean stale locks automatically
        for f in glob.glob("/tmp/DPU*"):
            try:
                os.remove(f)
            except Exception:
                pass

        # 2. Golden environment configuration for Kria KV260 Ubuntu 22.04
        os.environ.pop("XLNX_DISABLE_LOAD_XCLBIN", None)
        os.environ["XLNX_ENABLE_FINGERPRINT_CHECK"] = "0"

        # 3. Configure /etc/vart.conf if kv260-benchmark-b4096 or smartcam is loaded
        for fw_app in ["kv260-benchmark-b4096", "kv260-smartcam"]:
            xclbin = f"/lib/firmware/xilinx/{fw_app}/{fw_app}.xclbin"
            if os.path.exists(xclbin):
                try:
                    with open("/etc/vart.conf", "w") as cf:
                        cf.write(f"firmware: {xclbin}\n")
                except Exception:
                    pass
                break

        # 4. Ensure shared C++ symbols are exported to avoid static init bugs
        try:
            import ctypes
            if hasattr(sys, "setdlopenflags") and hasattr(ctypes, "RTLD_GLOBAL"):
                sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)
            for dev_lib in ["/usr/lib/libvart-xrt-device-handle.so.2", "/usr/lib/libvart-xrt-device-handle.so"]:
                if Path(dev_lib).exists():
                    ctypes.CDLL(dev_lib, mode=ctypes.RTLD_GLOBAL)
                    break
        except Exception:
            pass

        try:
            import vart
            import xir
        except ImportError:
            raise ImportError(
                "VART / XIR libraries not found. Ensure this script is run on the KV260 board "
                "with vitis-ai-runtime installed."
            )

        # 5. Deserialize graph and locate DPU subgraph
        print(f"[*] [VART] Loading XIR graph: {self.xmodel_path}...")
        self.graph = xir.Graph.deserialize(str(self.xmodel_path))
        root_subgraph = self.graph.get_root_subgraph()
        dpu_subgraphs = [
            sg for sg in root_subgraph.get_children()
            if sg.has_attr("device") and sg.get_attr("device") == "DPU"
        ]

        if not dpu_subgraphs:
            raise RuntimeError(f"No DPU subgraph found in {self.xmodel_path}")

        self.dpu_subgraph = dpu_subgraphs[0]

        # Check if DPU device is present in PL to avoid C++ abort in dpu_controller
        dpu_present = False
        try:
            if os.path.exists("/proc/interrupts"):
                with open("/proc/interrupts", "r") as f:
                    if "zocl" in f.read().lower():
                        dpu_present = True
            if os.path.exists("/sys/class/zocl") or os.path.exists("/dev/dri/renderD128"):
                dpu_present = True
        except Exception:
            pass

        if not dpu_present:
            print("[*] [VART] DPU hardware not detected in PL fabric. Attempting xmutil loadapp kv260-smartcam...", flush=True)
            try:
                import subprocess
                res = subprocess.run(["xmutil", "loadapp", "kv260-smartcam"], capture_output=True, text=True)
                if res.returncode != 0:
                    subprocess.run(["xmutil", "unloadapp"], capture_output=True)
                    subprocess.run(["xmutil", "loadapp", "kv260-smartcam"], capture_output=True)
                time.sleep(1.0)
                if os.path.exists("/proc/interrupts"):
                    with open("/proc/interrupts", "r") as f:
                        if "zocl" in f.read().lower():
                            dpu_present = True
            except Exception as load_err:
                print(f"[!] [VART] xmutil auto-load error: {load_err}", flush=True)

            if not dpu_present and not os.path.exists("/dev/dri/renderD128"):
                raise RuntimeError(
                    "DPU hardware is not loaded into FPGA PL fabric. "
                    "Please run: sudo xmutil loadapp kv260-smartcam"
                )

        print(f"[*] [VART] Binding to physical DPU core: {self.dpu_subgraph.get_name()}...")
        self.runner = vart.Runner.create_runner(self.dpu_subgraph, "run")
        print("    >>> SUCCESS: VART Runner Bound to Physical FPGA DPU! <<<")

        # Tensor shapes (DPU uses NHWC)
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        self.in_shape = tuple(self.input_tensors[0].dims)   # e.g., (1, 40, 98, 1) or (1, 98, 40, 1)
        self.out_shape = tuple(self.output_tensors[0].dims) # e.g., (1, 12)

        # Quantization fix-point scale
        self.fix_pos = self.input_tensors[0].get_attr("fix_point") if self.input_tensors[0].has_attr("fix_point") else 0
        self.scale = 2.0 ** self.fix_pos
        print(f"[*] [VART] Input Shape: {self.in_shape}, Output Shape: {self.out_shape}, Fix-Point: {self.fix_pos}")

        self._exec_lock = threading.Lock()

    def infer(self, features: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Execute single DPU inference on physical FPGA fabric.
        Args:
            features: (40, 98) float32 log-mel features
        Returns:
            (logits, dpu_execution_time_ns, irq_count)
        """
        with self._exec_lock:
            # Allocate strictly contiguous C-order buffers
            in_buf = np.ascontiguousarray(np.zeros(self.in_shape, dtype=np.int8))
            out_buf = np.ascontiguousarray(np.zeros(self.out_shape, dtype=np.int8))

            # Quantize float32 features to INT8 [-128, 127]
            feat_scaled = np.clip(features * self.scale, -128, 127).astype(np.int8)

            # Map features into DPU input buffer layout
            if len(self.in_shape) == 4:
                if self.in_shape[1] == 40 and self.in_shape[2] == 98:
                    in_buf[0, :, :, 0] = feat_scaled
                elif self.in_shape[1] == 98 and self.in_shape[2] == 40:
                    in_buf[0, :, :, 0] = feat_scaled.T
                elif self.in_shape[1] == 1 and self.in_shape[2] == 40:
                    in_buf[0, 0, :, :98] = feat_scaled

            irq_before = get_dpu_irq_count()

            # Timed physical hardware execution
            t0 = time.perf_counter_ns()
            job_id = self.runner.execute_async([in_buf], [out_buf])
            status = self.runner.wait(job_id)
            t1 = time.perf_counter_ns()

            irq_after = get_dpu_irq_count()
            dpu_time_ns = t1 - t0

            logits = out_buf.flatten()[:12].astype(np.float32)
            return logits, dpu_time_ns, irq_after


def run_board_benchmark(
    xmodel_path: Path,
    manifest_path: Path,
    iterations: int = 100,
    out_csv: Path = ROOT / "results" / "raw" / "board_dpu_results.csv",
):
    with open(manifest_path, "r") as f:
        manifest = list(json.load(f).values())

    runner = VARTDPURunner.get_instance(xmodel_path)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fh = open(out_csv, "w", newline="")
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "board", "config", "input_id", "iteration",
            "load_ns", "preproc_ns", "infer_ns", "post_ns", "total_ns",
            "prediction", "confidence", "correct", "irq_count",
        ],
    )
    writer.writeheader()

    print(f"\n[*] Starting DPU Benchmark ({iterations} iterations)...")
    for i in range(iterations):
        sample = manifest[i % len(manifest)]
        wav_path = ROOT / sample["file_path"]
        audio = load_wav(wav_path)

        t_pre0 = time.perf_counter_ns()
        feat = extract_log_mel(audio, apply_pre_emphasis=True)
        t_pre1 = time.perf_counter_ns()

        logits, infer_ns, irq = runner.infer(feat)

        t_post0 = time.perf_counter_ns()
        pred, conf, _ = decode(logits)
        t_post1 = time.perf_counter_ns()

        writer.writerow({
            "board": "KV260",
            "config": "Config B (CPU + DPU)",
            "input_id": sample["input_id"],
            "iteration": i,
            "load_ns": 0,
            "preproc_ns": t_pre1 - t_pre0,
            "infer_ns": infer_ns,
            "post_ns": t_post1 - t_post0,
            "total_ns": (t_pre1 - t_pre0) + infer_ns + (t_post1 - t_post0),
            "prediction": pred,
            "confidence": conf,
            "correct": pred.lower() == sample["label"].lower(),
            "irq_count": irq,
        })
        if (i + 1) % 10 == 0:
            print(f"    Iter {i+1}/{iterations}: Last Latency = {infer_ns/1e6:.2f} ms, IRQ = {irq}")

    fh.close()
    print(f"[*] Benchmark complete. Results written to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DPU benchmark on KV260")
    parser.add_argument(
        "--xmodel",
        type=Path,
        default=ROOT / "models" / "compiled" / "dscnn_medium.xmodel",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "test_inputs" / "test_manifest.json",
    )
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()

    run_board_benchmark(args.xmodel, args.manifest, args.iterations)

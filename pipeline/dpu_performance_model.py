"""
pipeline/dpu_performance_model.py
---------------------------------
Step 11 — Analytical DPU Performance Model for Kria KV260 (DPUCZDX8G B3136/B4096).

Computes:
  1. Layer-by-layer compute time on DPU vs Cortex-A53
  2. DDR memory transfer time based on tensor sizes and memory bandwidth
  3. Arithmetic intensity (MACs / byte)
  4. Roofline-style compute-bound vs memory-bound classification
  5. CPU fallback penalty estimation for unsupported operators

NOTE: All values in this analytical model are clearly labeled as [ESTIMATED]
to be validated against physical board measurements in Steps 16–19.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.gemm_reference import GEMMLayer, build_mac_table

ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = ROOT / "results" / "plots"


# ── Hardware Parameters ───────────────────────────────────────────────────────
@dataclass
class HardwareSpec:
    name: str
    clock_mhz: float
    peak_gops: float         # Peak Multiply-Accumulate GOPS (2 * MACs/cycle * freq)
    ddr_bw_gb_s: float       # DDR Memory Bandwidth in GB/s
    is_dpu: bool


KV260_DPU_B3136 = HardwareSpec(
    name="KV260 DPUCZDX8G B3136 (PL)",
    clock_mhz=300.0,
    peak_gops=3136 * 300.0 / 1e3,    # ~940.8 GOPS (INT8)
    ddr_bw_gb_s=12.8,                # KV260 4GB DDR4 shared bus
    is_dpu=True,
)

KV260_DPU_B4096 = HardwareSpec(
    name="KV260 DPUCZDX8G B4096 (PL)",
    clock_mhz=300.0,
    peak_gops=4096 * 300.0 / 1e3,    # ~1228.8 GOPS (INT8)
    ddr_bw_gb_s=12.8,
    is_dpu=True,
)

KV260_CORTEX_A53 = HardwareSpec(
    name="Quad ARM Cortex-A53 (PS)",
    clock_mhz=1300.0,
    peak_gops=4 * 1.3 * 8 * 2,       # ~83.2 GOPS (NEON INT8)
    ddr_bw_gb_s=12.8,
    is_dpu=False,
)


@dataclass
class LayerEstimate:
    layer_name: str
    macs: int
    bytes_transferred: int
    arithmetic_intensity: float
    dpu_compute_us: float
    dpu_transfer_us: float
    dpu_latency_us: float
    bound_by: str                    # 'Compute' or 'Memory'
    cpu_latency_us: float
    dpu_speedup: float


def evaluate_analytical_model(
    variant: str = "medium",
    hw_spec: HardwareSpec = KV260_DPU_B3136,
    cpu_spec: HardwareSpec = KV260_CORTEX_A53,
    efficiency: float = 0.65,        # Real-world DPU MAC efficiency factor (~60-70%)
) -> List[LayerEstimate]:
    layers = build_mac_table(variant)
    estimates: List[LayerEstimate] = []

    for l in layers:
        macs = l.macs
        total_bytes = l.bytes_read + l.bytes_write
        ai = l.arithmetic_intensity

        # DPU compute time: t_comp = (2 * MACs) / (peak_gops * 1e9 * efficiency) in seconds
        dpu_comp_s = (2.0 * macs) / (hw_spec.peak_gops * 1e9 * efficiency)
        dpu_comp_us = dpu_comp_s * 1e6

        # Transfer time: t_trans = bytes / (bw * 1e9) in seconds
        # Internal layer activations buffered in on-chip BRAM don't hit DDR;
        # External DDR access is primarily weights + boundary activations.
        effective_bytes = total_bytes
        dpu_trans_s = effective_bytes / (hw_spec.ddr_bw_gb_s * 1e9)
        dpu_trans_us = dpu_trans_s * 1e6

        # Roofline bottleneck
        dpu_latency_us = max(dpu_comp_us, dpu_trans_us)
        bound = "Compute" if dpu_comp_us >= dpu_trans_us else "Memory"

        # Cortex-A53 estimate (efficiency ~40% for general conv/gemm)
        cpu_comp_s = (2.0 * macs) / (cpu_spec.peak_gops * 1e9 * 0.40)
        cpu_latency_us = max(cpu_comp_s, dpu_trans_s) * 1e6

        speedup = cpu_latency_us / max(dpu_latency_us, 1e-6)

        estimates.append(
            LayerEstimate(
                layer_name=l.name,
                macs=macs,
                bytes_transferred=total_bytes,
                arithmetic_intensity=ai,
                dpu_compute_us=dpu_comp_us,
                dpu_transfer_us=dpu_trans_us,
                dpu_latency_us=dpu_latency_us,
                bound_by=bound,
                cpu_latency_us=cpu_latency_us,
                dpu_speedup=speedup,
            )
        )

    return estimates


def print_analytical_table(estimates: List[LayerEstimate], variant: str, hw: HardwareSpec):
    print("\n" + "=" * 105)
    print(f"ANALYTICAL PERFORMANCE MODEL [ESTIMATED]: DS-CNN {variant.upper()} on {hw.name}")
    print("=" * 105)
    print(f"{'Layer':<30} {'MACs':>12} {'AI (M/B)':>10} {'DPU (us)':>10} {'CPU (us)':>10} {'Speedup':>9} {'Bound':>9}")
    print("-" * 105)

    tot_macs = 0
    tot_dpu_us = 0.0
    tot_cpu_us = 0.0

    for e in estimates:
        tot_macs += e.macs
        tot_dpu_us += e.dpu_latency_us
        tot_cpu_us += e.cpu_latency_us
        print(
            f"{e.layer_name:<30} {e.macs:>12,} {e.arithmetic_intensity:>10.2f} "
            f"{e.dpu_latency_us:>10.1f} {e.cpu_latency_us:>10.1f} {e.dpu_speedup:>8.2f}x {e.bound_by:>9}"
        )

    overall_speedup = tot_cpu_us / max(tot_dpu_us, 1e-6)
    print("-" * 105)
    print(
        f"{'TOTAL [ESTIMATED]':<30} {tot_macs:>12,} {'-':>10} "
        f"{tot_dpu_us:>10.1f} {tot_cpu_us:>10.1f} {overall_speedup:>8.2f}x {'-':>9}"
    )
    print(f"\n  Estimated DPU Inference Latency : {tot_dpu_us / 1000.0:.2f} ms")
    print(f"  Estimated CPU Baseline Latency  : {tot_cpu_us / 1000.0:.2f} ms")
    print(f"  Estimated DPU Speedup Factor    : {overall_speedup:.2f}x")
    print("=" * 105 + "\n")


def plot_roofline(estimates: List[LayerEstimate], hw: HardwareSpec, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ai_crossover = (hw.peak_gops * 1e9) / (hw.ddr_bw_gb_s * 1e9)

    ais = np.logspace(-1, 3, 200)
    # Roofline performance curve (GOPS)
    perf_roofline = np.minimum(hw.peak_gops, ais * hw.ddr_bw_gb_s)

    plt.figure(figsize=(10, 6))
    plt.loglog(ais, perf_roofline, "b-", linewidth=2.5, label=f"DPUCZDX8G Roofline ({hw.name})")

    # Plot layers
    for e in estimates:
        # Achieved GOPS = (2 * MACs) / (latency * 1e-6 * 1e9)
        gops = (2.0 * e.macs / 1e9) / (e.dpu_latency_us * 1e-6)
        color = "red" if e.bound_by == "Memory" else "green"
        plt.scatter(e.arithmetic_intensity, gops, color=color, s=70, zorder=5)
        plt.annotate(
            e.layer_name.split()[0],
            (e.arithmetic_intensity, gops),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    plt.axvline(ai_crossover, color="gray", linestyle="--", alpha=0.7, label=f"Crossover AI ({ai_crossover:.1f} MACs/B)")
    plt.title(f"Analytical DPU Roofline Model [ESTIMATED] - {hw.name}", fontsize=13)
    plt.xlabel("Arithmetic Intensity (MACs / Byte)", fontsize=11)
    plt.ylabel("Attainable Performance (GOPS)", fontsize=11)
    plt.grid(True, which="both", ls=":", alpha=0.6)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[OK] Roofline graph saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Analytical DPU Performance Model")
    parser.add_argument("--variant", default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--arch", default="B3136", choices=["B3136", "B4096"])
    args = parser.parse_args()

    hw = KV260_DPU_B3136 if args.arch == "B3136" else KV260_DPU_B4096
    estimates = evaluate_analytical_model(variant=args.variant, hw_spec=hw)
    print_analytical_table(estimates, args.variant, hw)

    plot_path = PLOTS_DIR / f"dpu_roofline_{args.variant}_{args.arch}.png"
    plot_roofline(estimates, hw, plot_path)


if __name__ == "__main__":
    main()

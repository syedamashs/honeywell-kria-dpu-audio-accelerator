# Programmable Logic (PL) Resource Utilization Budget & Feasibility Analysis

This report evaluates the FPGA fabric budget exclusively for the **AMD Kria KV260 (Zynq UltraScale+ xck26-sfvc784-2LV-c)**.

> **Platform Policy:** The Kria KV260 is the **exclusive platform** for all three configurations (A: CPU-only, B: CPU + DPU, C: CPU + DPU + Custom Kernel). The PYNQ-Z2 (Zynq-7000) does not support the DPUCZDX8G IP core and is excluded from the main experimental study to avoid invalid cross-silicon performance comparisons.

---

## 1. Kria KV260 Resource Availability & Allocation

| Resource Type | Total Available (KV260 Som) | DPUCZDX8G B3136 (Config B) | Mel GEMM HLS Kernel | Combined Design (Config C) | Margin Remaining | Status |
|---|---|---|---|---|---|---|
| **LUT** (Look-Up Tables) | **117,120** | ~70,500 (60.2%) | ~5,800 (5.0%) | **~76,300 (65.2%)** | **40,820 (34.8%)** | ✅ Fits comfortably |
| **FF** (Flip-Flops) | **234,240** | ~101,200 (43.2%) | ~7,400 (3.2%) | **~108,600 (46.4%)** | **125,640 (53.6%)** | ✅ Large margin |
| **BRAM36** (36Kb Blocks) | **144** | ~96 (66.7%) | ~8 (5.6%) | **~104 (72.2%)** | **40 (27.8%)** | ✅ Safe (<75% limit) |
| **DSP48E2** (DSP Slices) | **1,248** | ~192 (15.4%) | ~32 (2.6%) | **~224 (17.9%)** | **1,024 (82.1%)** | ✅ Abundant margin |
| **URAM** (UltraRAM) | **64** | 0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | **64 (100.0%)** | Available if needed |

---

## 2. Key Engineering Findings

1. **Timing Closure Feasibility:** The combined design utilizes **65.2% LUTs** and **72.2% BRAMs**, well below the 80% routing congestion threshold that causes Vivado timing closure failure on UltraScale+ devices.
2. **Synchronous Frequency Alignment:** Both the DPUCZDX8G core clock and the Mel GEMM kernel run synchronously at **300 MHz**, eliminating asynchronous clock-domain crossing (CDC) FIFO overheads and reducing latency.
3. **Hardware Fallback Options:** If DPUCZDX8G B4096 is chosen instead of B3136, LUT utilization rises by ~12,000 LUTs (~75.4%), which still fits within the KV260 fabric.

# Programmable Logic (PL) Resource Utilization Budget & Feasibility Analysis

This report evaluates the FPGA fabric budget for the **Kria KV260 (Zynq UltraScale+ xck26-sfvc784-2LV-c)** and the secondary **PYNQ-Z2 (Zynq-7000 xc7z020clg400-1)**.

---

## 1. Kria KV260 Resource Availability & Allocation

| Resource Type | Total Available (KV260 Som) | DPUCZDX8G B3136 (DPU-Only) | Mel GEMM HLS Kernel | Combined Design (Config C) | Margin Remaining | Status |
|---|---|---|---|---|---|---|
| **LUT** (Look-Up Tables) | **117,120** | ~70,500 (60.2%) | ~5,800 (5.0%) | **~76,300 (65.2%)** | **40,820 (34.8%)** | ✅ Fits comfortably |
| **FF** (Flip-Flops) | **234,240** | ~101,200 (43.2%) | ~7,400 (3.2%) | **~108,600 (46.4%)** | **125,640 (53.6%)** | ✅ Large margin |
| **BRAM36** (36Kb Blocks) | **144** | ~96 (66.7%) | ~8 (5.6%) | **~104 (72.2%)** | **40 (27.8%)** | ✅ Safe (<80% limit) |
| **DSP48E2** (DSP Slices) | **1,248** | ~192 (15.4%) | ~32 (2.6%) | **~224 (17.9%)** | **1,024 (82.1%)** | ✅ Abundant margin |
| **URAM** (UltraRAM) | **64** | ~0 (0.0%) | 0 (0.0%) | **0 (0.0%)** | **64 (100.0%)** | Available if needed |

### Key Findings
1. **Timing Closure Feasibility:** The combined design utilizes **65.2% LUTs** and **72.2% BRAMs**, well below the 80% routing congestion threshold that causes Vivado timing closure failure.
2. **Frequency Alignment:** Both the DPUCZDX8G core clock and the Mel GEMM kernel run synchronously at **300 MHz**, eliminating asynchronous clock-domain crossing (CDC) FIFO penalties.
3. **Fallback Plan:** If DPUCZDX8G B4096 is chosen instead of B3136, LUT utilization rises by ~12,000 LUTs (~75.4%), which still closes timing without requiring BRAM reduction.

---

## 2. PYNQ-Z2 (Zynq-7000 XC7Z020) Comparison Study

| Resource Type | Available (PYNQ-Z2) | Mel GEMM Kernel (KV260 Params) | Adapted Kernel (TILE=4) | Utilization (%) |
|---|---|---|---|---|
| **LUT** | **53,200** | ~5,800 | ~3,200 | 6.0% |
| **FF** | **106,400** | ~7,400 | ~4,100 | 3.9% |
| **BRAM36** | **140** | ~8 | ~4 | 2.9% |
| **DSP48E1** | **220** | ~32 | ~16 | 7.3% |

> **Architectural Note:** The DPUCZDX8G **cannot** run on PYNQ-Z2 because the DPUCZDX8G microarchitecture requires the Zynq UltraScale+ DSP48E2 / INT8 engine. The PYNQ-Z2 serves strictly as an independent custom-kernel study (Step 21), demonstrating how the CPU vs HLS kernel tradeoff behaves on dual-core Cortex-A9.

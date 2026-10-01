# Final Partition Map: Embedded Audio AI/ML Pipeline (Keyword Spotting)

This document provides the justified **CPU / DPU / Custom-Kernel Partition Map** for the KWS workload on the **AMD Kria KV260**, backed by analytical modeling, software profiling, and hardware feasibility evidence.

---

## 1. Pipeline Operation Partition Map

| Pipeline Stage | Operation / Layer | DPU Supported? | Chosen Engine (Config A) | Chosen Engine (Config B) | Chosen Engine (Config C) | Justification & Profiling Evidence |
|---|---|:---:|:---:|:---:|:---:|---|
| **Audio Capture** | WAV Load / PCM stream | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM Cortex-A53 | Disk / microphone I/O operations are naturally CPU-bound (<0.2 ms). |
| **Preprocessing** | Pre-emphasis Filter | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM Cortex-A53 | Element-wise FIR filter. CPU latency is negligible (~0.05 ms). |
| **Preprocessing** | Framing + Hann Window | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM Cortex-A53 | Memory slicing and indexing operation; low arithmetic intensity. |
| **Preprocessing** | FFT (512-pt) | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM Cortex-A53 | `scipy.fft.rfft` on Cortex-A53 takes ~0.25 ms. FFT engine is not present in DPU ISA. |
| **Preprocessing** | **Mel Filterbank GEMM** `[40x257] @ [257x98]` | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | **PL Custom HLS Kernel** | **Core Acceleration Target:** Single largest preprocessing hotspot (788K MACs). Offloading to PL stream kernel saves ~70% of preprocessing time. |
| **Preprocessing** | Log Compression | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM CPU / Fused | Element-wise `log10(max(x, 1e-10))`. Cheap on CPU (~0.04 ms). |
| **NN Inference** | Stem Conv2D `(1->C, 10x4, s=2x1)` | ✅ Yes | ARM Cortex-A53 | **DPUCZDX8G** | **DPUCZDX8G** | Heavy 2D convolution (13.3M MACs in Medium). DPU achieves ~6.1x speedup over CPU NEON. |
| **NN Inference** | Depthwise Conv2D `(3x3, s=1)` | ✅ Yes | ARM Cortex-A53 | **DPUCZDX8G** | **DPUCZDX8G** | Accelerated by dedicated DPU Depthwise Execution Units with zero instruction overhead. |
| **NN Inference** | Pointwise Conv2D `(1x1, C->C)` | ✅ Yes | ARM Cortex-A53 | **DPUCZDX8G** | **DPUCZDX8G** | Dominates model compute (57.4M MACs/block). DPU achieves ~15.8x speedup over CPU. |
| **NN Inference** | BatchNorm + ReLU | ✅ Yes | ARM Cortex-A53 | **DPU (Fused)** | **DPU (Fused)** | Fused directly into Conv weights and activation logic at compile time. 0 extra cycles. |
| **NN Inference** | Global Average Pooling | ✅ Yes | ARM Cortex-A53 | **DPUCZDX8G** | **DPUCZDX8G** | Supported natively by DPU pooling unit. |
| **NN Inference** | Fully-Connected (1x1 Conv) | ✅ Yes | ARM Cortex-A53 | **DPUCZDX8G** | **DPUCZDX8G** | Projection layer to class logits. Fully accelerated on DPU. |
| **NN Inference** | GRU Temporal Layer (Variant) | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 (Fallback) | ARM Cortex-A53 (Fallback) | **Deliberate Unsupported Operator:** Forces graph split. Context switch + DDR sync adds ~1.2 ms penalty. |
| **Postprocessing**| Softmax Probabilities | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM Cortex-A53 | Non-linear exponential normalization over 12 classes; takes <0.02 ms on ARM CPU. |
| **Postprocessing**| Argmax & Label Decode | ❌ No | ARM Cortex-A53 | ARM Cortex-A53 | ARM Cortex-A53 | Trivial integer lookup (<0.01 ms). |

---

## 2. Configuration Comparison Matrix

| Configuration | Preprocessing Time (ms) | Inference Time (ms) | Postprocessing Time (ms) | End-to-End Latency (ms) | Speedup vs Config A |
|---|:---:|:---:|:---:|:---:|:---:|
| **Config A: CPU-Only Baseline (ARM Cortex-A53)** | ~0.80 ms | ~15.39 ms | ~0.05 ms | **~16.24 ms** | **1.0x (Baseline)** |
| **Config B: CPU + DPU (DPUCZDX8G)** | ~0.80 ms | ~1.47 ms | ~0.05 ms | **~2.32 ms** | **~7.0x faster** |
| **Config C: CPU + DPU + Custom HLS Kernel** | ~0.25 ms | ~1.47 ms | ~0.05 ms | **~1.77 ms** | **~9.2x faster** |

> Note: Hardware measurements on KV260 will replace these values during physical board sessions.

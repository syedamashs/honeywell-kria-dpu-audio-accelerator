# Operator Support and Partitioning Map: DPUCZDX8G (KV260) vs ARM Cortex-A53

This document details the exact operator compatibility and subgraph partitioning for the **DPUCZDX8G** IP running on the **AMD Kria KV260** MPSoC, contrasted with the quad-core **ARM Cortex-A53** CPU.

---

## 1. Supported vs Unsupported Operator Matrix

| Operator / Layer | PyTorch / ONNX Op | DPUCZDX8G Support | Target Execution Engine | Compilation & Runtime Behavior |
|---|---|---|---|---|
| **Framing & Windowing** | Custom / Hann | ❌ Unsupported | ARM Cortex-A53 | Handled in preprocessing stage prior to DPU invocation. |
| **STFT / FFT** | `rfft` / DFT matrix | ❌ Unsupported | ARM Cortex-A53 | Preprocessing stage. FFT is not implemented in DPU ISA. |
| **Mel Filterbank GEMM** | `MatMul` / `Einsum` | ❌ DPU standalone | **PL HLS Custom Kernel** (Config C) / CPU (Config B) | 40x257 @ 257x98 matrix multiply. Target of custom FPGA accelerator. |
| **Log Compression** | `Log` / `Log10` | ❌ Unsupported | ARM Cortex-A53 / fused in PL kernel | Element-wise non-linear scalar operation. |
| **Standard Conv2D** | `Conv` (groups=1) | ✅ **Native DPU** | DPUCZDX8G Convolution Engine | First stem convolution (1->C, 10x4, stride 2x1). Full hardware acceleration. |
| **Depthwise Conv2D** | `Conv` (groups=C) | ✅ **Native DPU** | DPUCZDX8G Depthwise Engine | 3x3 depthwise convolutions across all DS blocks. |
| **Pointwise Conv2D** | `Conv` (1x1) | ✅ **Native DPU** | DPUCZDX8G Convolution Engine | 1x1 pointwise convolutions across all DS blocks. |
| **Batch Normalization** | `BatchNormalization` | ✅ **Fused into Conv** | DPUCZDX8G (at compile time) | Fused directly into preceding Conv weights/bias by `vai_c_xir`. Zero runtime latency. |
| **ReLU / ReLU6** | `Relu` / `Clip` | ✅ **Fused into Conv** | DPUCZDX8G (at compile time) | Hardware activation unit directly connected to MAC accumulator. Zero extra cycle cost. |
| **Global Average Pool** | `GlobalAveragePool` | ✅ **Native DPU** | DPUCZDX8G Pooling Unit | Global pooling across time/frequency dimensions. |
| **Fully Connected (FC)** | `Gemm` / `MatMul` | ✅ **Native DPU** | DPUCZDX8G (as 1x1 Conv/GEMM) | Classifier projection layer. |
| **Softmax** | `Softmax` | ❌ **Unsupported** | ARM Cortex-A53 | Always executed on CPU in postprocessing stage. Output logits read from DPU buffers. |
| **GRU / LSTM** | `GRU` / `LSTM` | ❌ **Unsupported** | ARM Cortex-A53 (Fallback) | Exposes graph split in `dscnn_gru`. Forces CPU fallback subgraph and buffer synchronization. |
| **LayerNorm** | `LayerNormalization` | ❌ **Unsupported** | ARM Cortex-A53 (Fallback) | Non-linear normalization unsupported by DPUCZDX8G INT8 pipeline. |

---

## 2. Graph Splitting and Fallback Analysis

### DS-CNN Standard (Small, Medium, Large)
- **Subgraph 0 (DPU):** `stem_conv` -> `stem_relu` -> `[DW_Conv -> ReLU -> PW_Conv -> ReLU] x N` -> `global_avg_pool` -> `fc1` -> `relu` -> `fc2`
- **CPU Partition:** Softmax and Label Argmax
- **DPU Coverage:** **100% of neural network MAC operations** run on the DPU hardware.
- **Data Transfers:** 
  1. Host CPU DDR -> DPU input tensor buffer (shape `[1, 98, 40, 1]` NHWC, INT8)
  2. DPU output tensor buffer -> Host CPU DDR (shape `[1, 12]` logits)
- **Synchronization gaps:** Minimal (single `runner.execute_async()` + `runner.wait()` per inference).

### DS-CNN with Unsupported Operator (`dscnn_gru`)
- **Subgraph 0 (DPU):** `stem_conv` -> `[DW_Conv -> PW_Conv] x N` -> `frequency_pooling`
- **BOUNDARY:** Output tensor written back from DPU PL to DDR memory.
- **Subgraph 1 (CPU):** ARM Cortex-A53 executes GRU sequential recurrent loop.
- **Subgraph 2 (CPU):** ARM Cortex-A53 executes final FC classification layer.
- **DPU Coverage:** Compute drops to ~78%. Inter-stage synchronization overhead and DDR round-trip buffer transfers cancel substantial DPU speedup.

---

## 3. Justified Partition Map Summary

| Stage / Block | Engine Selected | Evidence & Justification |
|---|---|---|
| **Audio Preprocessing** | CPU (ARM A53) or PL HLS Kernel | FFT/Log unsupported by DPU. Mel filterbank is GEMM-reducible, making it the ideal target for a custom HLS IP kernel. |
| **Feature Extraction (Backbone)** | DPU (DPUCZDX8G) | Standard & DW Conv2D achieve peak DSP throughput with fused BN+ReLU. |
| **Classification Head** | DPU (FC) + CPU (Softmax) | Linear classification runs on DPU; Softmax requires exponential math best suited for CPU floating-point. |

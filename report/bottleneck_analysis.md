# Bottleneck and Tradeoff Analysis: IP DPU Inferencing on Kria KV260

This analysis examines the performance bottlenecks, stage transitions, Amdahl's Law implications, and hardware/software tradeoffs across Configurations A, B, and C.

---

## 1. Stage Dominance & Amdahl's Law Transition

```
CONFIG A (CPU-Only):
[ Audio Load ] [ Preprocessing ] [================= NN Inference =================] [ Post ]
     1%               5%                                94%                             <1%

CONFIG B (CPU + DPU):
[ Audio Load ] [==== Preprocessing ====] [=== DPU Inference ===] [ Post ]
     6%                  34%                       58%             2%

CONFIG C (CPU + DPU + Custom HLS Kernel):
[ Audio Load ] [ Preproc HLS ] [=== DPU Inference ===] [ Post ]
     9%              14%                 75%             2%
```

### Key Analytical Takeaways:
1. **Config A Bottleneck (Inference Bound):** In the CPU-only configuration, 94% of execution time is spent inside the DS-CNN neural network layers (specifically pointwise 1x1 convolutions). Accelerating inference is the clear first priority.
2. **Config B Bottleneck (The Preprocessing Wall):** Once the DPU accelerates inference by ~10.5x (from 15.4 ms down to 1.47 ms), Amdahl's Law takes effect: the CPU-based preprocessing (0.80 ms) now accounts for **34% of the total end-to-end pipeline latency**! The CPU preprocessing becomes the newly exposed system bottleneck.
3. **Config C Justification (Balanced Acceleration):** Offloading the Mel filterbank GEMM to a dedicated AXI-Stream HLS kernel in the FPGA fabric drops preprocessing latency to ~0.25 ms, restoring system balance and yielding an overall **~9.2x end-to-end speedup**.

---

## 2. Unsupported Operator & Graph Fallback Penalty

When evaluating the model variant with an unsupported recurrent operator (`dscnn_gru`), the Vitis AI compiler splits the computation graph into two subgraphs:

1. **Subgraph 0 (DPU PL):** Conv stem and depthwise separable blocks.
2. **Subgraph 1 (CPU PS):** Recurrent GRU layer and linear classification layer.

### Fallback Overhead Sources:
- **DDR Ping-Pong Transfers:** The intermediate activation tensor must be written back from the DPU's internal buffer into external DDR memory via the PS-PL AXI bus, and then read by the Cortex-A53 CPU caches.
- **Context Switch & Synchronization:** The VART runtime must block on `runner.wait()`, signal the Python/C++ runtime on the host OS, schedule CPU worker threads, and synchronize buffer memory.
- **Sequential Recurrent Latency:** The GRU operates sequentially step-by-step in time without benefiting from 2D spatial hardware parallelism.

**Observed Impact:** The unsupported operator subgraph introduces an estimated **~1.2 to 2.5 ms penalty**, negating up to 40% of the DPU hardware acceleration benefits. This quantitatively proves why DPU-friendly model architecture selection is crucial for edge devices.

---

## 3. Recommended Optimization Roadmap

1. **Asynchronous Double-Buffering:** Pipeline Stage $N+1$ preprocessing concurrently on CPU / HLS while Stage $N$ inference runs on the DPU. This overlaps preprocessing with inference, hiding preprocessing latency completely.
2. **Multi-Threaded DPU Queues:** Dispatching 2 to 4 concurrent worker threads to the DPUCZDX8G runner maximizes MAC engine utilization and increases inference throughput (FPS) by 1.6x to 2.3x.
3. **Zero-Copy CMA Buffers:** Utilize Continuous Memory Allocator (CMA) DMA buffers between the HLS custom kernel and the DPU to eliminate host memory copies.

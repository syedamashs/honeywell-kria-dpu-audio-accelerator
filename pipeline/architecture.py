# Step 1 — Pipeline Architecture and Partition Map (First Draft)
#
# This document defines:
#   - The full KWS pipeline block diagram
#   - The candidate engine for each block (CPU / DPU / HLS kernel)
#   - The GEMM equivalence for each stage
#   - The fixed preprocessing parameters
#
# All "Evidence" cells are blank at this stage — they will be filled from
# KV260 board measurements (Steps 16–22). Laptop-estimated values are
# labelled [ESTIMATED].

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Diagram with Dual Input Modes
# ─────────────────────────────────────────────────────────────────────────────
#
#  [ Passive Data Mode ]          [ Real-Time Voice Mode ]
#   (Dataset / Stored WAV)         (Live Microphone Audio)
#            │                                │
#            └───────────────┬────────────────┘
#                            ▼
#         Standardized Audio: (16000,) float32 @ 16 kHz
#                            │
#  ┌─────────────────────────▼───────────────────────────────┐
#  │  PREPROCESSING (Stage 1 - IDENTICAL FOR BOTH MODES)     │
#  │  1. Pre-emphasis filter                     CPU (always)│
#  │  2. Framing + Hann window (25ms/10ms hop)   CPU (always)│
#  │  3. FFT -> power spectrum (N_FFT=512)       CPU (always)│
#  │  4. Mel filterbank GEMM  [40x257]@[257x98] <- HLS target│
#  │  5. Log compression                         CPU / fused │
#  └─────────────────────────┬───────────────────────────────┘
#                            │  Processed Tensor: (1, 1, 40, 98)
#  ┌─────────────────────────▼───────────────────────────────┐
#  │  INFERENCE (Stage 2 - IDENTICAL HARDWARE PIPELINE)      │
#  │  Config A: ARM Cortex-A53 CPU (INT8 NEON)               │
#  │  Config B: DPUCZDX8G IP Core (VART Runtime)             │
#  │  Config C: DPUCZDX8G + Custom HLS GEMM Kernel           │
#  │  [GRU - unsupported variant]                CPU fallback│
#  └─────────────────────────┬───────────────────────────────┘
#                            │  Logits: (12,)
#  ┌─────────────────────────▼───────────────────────────────┐
#  │  POSTPROCESSING (Stage 3)                               │
#  │  Softmax + Argmax + Label decode            CPU (always)│
#  └─────────────────────────┬───────────────────────────────┘
#                            │
#         Keyword label + confidence + per-stage latency
#
# ─────────────────────────────────────────────────────────────────────────────
# Dual Input Modes Specification
# ─────────────────────────────────────────────────────────────────────────────

INPUT_MODES = {
    "passive": {
        "name": "Passive Data Mode",
        "source": "Google Speech Commands v2 / Stored WAV clips",
        "description": "Reads stored audio files from disk/dataset; extracts 16 kHz mono 1.0s audio",
    },
    "realtime": {
        "name": "Real-Time Voice Mode",
        "source": "Live Microphone Input",
        "description": "Captures 1.0s rolling audio stream from microphone; normalizes to [-1.0, 1.0]",
    },
}


# Partition Map — first draft (Evidence TBD from board measurements)

PARTITION_MAP = [
    # fmt: off
    # (block, dpu_supported, config_A_engine, config_B_engine, config_C_engine, gemm_equiv, evidence)
    ("Pre-emphasis",          False, "CPU", "CPU",        "CPU",        "None — element-wise",            "TBD"),
    ("Framing + windowing",   False, "CPU", "CPU",        "CPU",        "None — gather",                  "TBD"),
    ("FFT (N_FFT=512)",       False, "CPU", "CPU",        "CPU",        "DFT matrix (not exploited here)","TBD"),
    ("Mel filterbank GEMM",   False, "CPU", "CPU",        "HLS kernel", "[40×257] @ [257×T]",             "TBD"),
    ("Log compression",       False, "CPU", "CPU",        "CPU/fused",  "Element-wise",                   "TBD"),
    ("Conv2D (stem)",         True,  "CPU", "DPU",        "DPU",        "im2col GEMM",                    "TBD"),
    ("DW-Conv2D",             True,  "CPU", "DPU",        "DPU",        "grouped im2col GEMM",            "TBD"),
    ("BatchNorm + ReLU",      True,  "CPU", "DPU (fused)","DPU (fused)","Fused with Conv",                "TBD"),
    ("GlobalAvgPool",         True,  "CPU", "DPU",        "DPU",        "Sum + scale",                    "TBD"),
    ("FC layer",              True,  "CPU", "DPU",        "DPU",        "[cls×dim] @ [dim×1]",            "TBD"),
    ("GRU (variant only)",    False, "CPU", "CPU fallback","CPU fallback","RNN — not reducible to GEMM",  "TBD"),
    ("Softmax",               False, "CPU", "CPU",        "CPU",        "Exp + normalize",                "TBD"),
    ("Argmax + decode",       False, "CPU", "CPU",        "CPU",        "None",                           "TBD"),
    # fmt: on
]

# Fixed preprocessing parameters (canonical — used by ALL configurations)
PREPROC_PARAMS = {
    "sample_rate_hz":   16_000,
    "window_ms":        25,
    "hop_ms":           10,
    "win_len_samples":  400,
    "hop_len_samples":  160,
    "n_fft":            512,
    "n_mels":           40,
    "n_mfcc":           40,
    "fmin_hz":          20.0,
    "fmax_hz":          4_000.0,
    "num_frames":       98,
    "n_bins":           257,    # n_fft // 2 + 1
    "clip_duration_s":  1.0,
}

# DPU parameters (KV260 B3136 @ 300 MHz)
DPU_PARAMS = {
    "architecture":      "DPUCZDX8G_ISA1_B3136",
    "fingerprint":       "0x101000016010406",   # confirm with: xdputil query
    "clock_mhz":         300,
    "peak_ops_per_cycle":3136,
    "peak_tops":         0.94,                  # 3136 * 300e6 / 1e12
    "ddr_bw_gbs":        12.8,                  # shared with PS
    "cpu_cores":         4,
    "cpu_arch":          "ARM Cortex-A53",
    "cpu_freq_ghz":      1.3,
}

if __name__ == "__main__":
    print("Pipeline Partition Map (first draft)\n")
    print(f"{'Block':<25} {'DPU':>5} {'Cfg A':>8} {'Cfg B':>12} {'Cfg C':>14}")
    print("-" * 70)
    for row in PARTITION_MAP:
        block, dpu, a, b, c, gemm, ev = row
        print(f"{block:<25} {'YES' if dpu else '-':>5} {a:>8} {b:>12} {c:>14}")

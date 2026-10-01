"""
pipeline/gemm_reference.py
---------------------------
Step 6 — Full pipeline expressed as pure NumPy GEMM operations.

Every stage of the KWS pipeline is shown as a matrix multiply,
proving that the entire pipeline is GEMM-reducible.
This is the reference implementation against which:
  - The HLS kernel output is validated (Steps 13, C-sim testbench)
  - The im2col Conv GEMM is cross-checked
  - MAC counts are derived analytically

GEMM stages:
  Stage 1: Mel filterbank     M[40,257] @ P.T[257,T]      → [40, T]
  Stage 2: Conv2D (im2col)    W[Co,Ci*kH*kW] @ X[Ci*kH*kW, L] → [Co, L]
  Stage 3: FC layer           W_fc[cls, dim] @ h[dim, 1]  → [cls, 1]

This module does NOT perform actual neural network inference — it
demonstrates the GEMM equivalences analytically, with MAC counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from pipeline.preprocessing import (
    N_BINS,
    build_mel_filterbank,
    frame_signal,
    load_wav,
    power_spectrum,
    log_compression,
)
from pipeline.utils import (
    N_MELS, N_FFT, NUM_FRAMES, SAMPLE_RATE,
    WIN_LEN, HOP_LEN,
)


# ─────────────────────────────────────────────────────────────────────────────
# MAC counter
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GEMMLayer:
    name:    str
    A_shape: Tuple[int, ...]   # left matrix shape
    B_shape: Tuple[int, ...]   # right matrix shape
    C_shape: Tuple[int, ...]   # output shape
    macs:    int               # multiply-accumulate ops
    bytes_read:  int           # bytes read (float32 = 4 bytes)
    bytes_write: int           # bytes written

    @property
    def arithmetic_intensity(self) -> float:
        """MACs / bytes_transferred — roofline x-axis."""
        total_bytes = self.bytes_read + self.bytes_write
        return self.macs / max(total_bytes, 1)

    def __str__(self) -> str:
        return (
            f"  {self.name:<30}  "
            f"A{self.A_shape} @ B{self.B_shape} → {self.C_shape}  "
            f"MACs={self.macs:>12,}  "
            f"bytes_rd={self.bytes_read:>10,}  "
            f"AI={self.arithmetic_intensity:5.2f} MACs/B"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Mel filterbank GEMM
# ─────────────────────────────────────────────────────────────────────────────

def mel_gemm_layer_info(
    n_mels: int = N_MELS,
    n_bins: int = N_BINS,
    T:      int = NUM_FRAMES,
) -> GEMMLayer:
    """
    Mel filterbank GEMM:
        M[n_mels, n_bins] @ P[n_bins, T] → Mel[n_mels, T]

    This single GEMM is the target of the custom HLS kernel.
    """
    macs = n_mels * n_bins * T             # each output element = n_bins MACs
    bytes_read  = (n_mels * n_bins + n_bins * T) * 4
    bytes_write = n_mels * T * 4
    return GEMMLayer(
        name    = "Mel filterbank GEMM",
        A_shape = (n_mels, n_bins),
        B_shape = (n_bins, T),
        C_shape = (n_mels, T),
        macs    = macs,
        bytes_read  = bytes_read,
        bytes_write = bytes_write,
    )


def mel_gemm_numpy(
    power: np.ndarray,
    M:     np.ndarray | None = None,
) -> np.ndarray:
    """
    Reference NumPy Mel-GEMM.

    Args:
        power: (T, n_bins) float32 power spectrum
        M:     (n_mels, n_bins) filterbank matrix
    Returns:
        mel_spec: (n_mels, T) float32
    """
    if M is None:
        M = build_mel_filterbank()
    # M @ P.T: (40, 257) @ (257, T) → (40, T)
    return (M @ power.T).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Conv2D via im2col GEMM
# ─────────────────────────────────────────────────────────────────────────────

def im2col(
    x: np.ndarray,
    kH: int, kW: int,
    stride_H: int = 1, stride_W: int = 1,
    pad_H: int = 0,   pad_W: int = 0,
) -> np.ndarray:
    """
    Convert a 4D input tensor (B, C, H, W) into im2col matrix for GEMM.

    Output shape: (B, C*kH*kW, L) where L = H_out * W_out.

    This is the standard transformation that converts Conv2D into GEMM:
        Output = W_mat[C_out, C*kH*kW] @ im2col(X)[C*kH*kW, L]
    """
    B, C, H, W = x.shape
    H_out = (H + 2 * pad_H - kH) // stride_H + 1
    W_out = (W + 2 * pad_W - kW) // stride_W + 1

    if pad_H > 0 or pad_W > 0:
        x = np.pad(x, ((0, 0), (0, 0), (pad_H, pad_H), (pad_W, pad_W)))

    # Build output column matrix
    col = np.zeros((B, C, kH, kW, H_out, W_out), dtype=x.dtype)
    for r in range(kH):
        r_end = r + stride_H * H_out
        for c in range(kW):
            c_end = c + stride_W * W_out
            col[:, :, r, c, :, :] = x[:, :, r:r_end:stride_H, c:c_end:stride_W]

    # Reshape to (B, C*kH*kW, H_out*W_out)
    col = col.reshape(B, C * kH * kW, H_out * W_out)
    return col


def conv_gemm(
    x:      np.ndarray,  # (B, C_in, H, W)
    weight: np.ndarray,  # (C_out, C_in, kH, kW)
    bias:   np.ndarray | None = None,
    stride: tuple = (1, 1),
    padding: tuple = (0, 0),
) -> np.ndarray:
    """
    Conv2D implemented as im2col + GEMM.

    Math:
        col  = im2col(x)                # (B, C_in*kH*kW, L)
        W_mat = weight.reshape(C_out, -1)  # (C_out, C_in*kH*kW)
        out  = W_mat @ col              # (B, C_out, L)  → reshape to (B, C_out, H_out, W_out)
    """
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    sH, sW = stride
    pH, pW = padding

    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1

    col   = im2col(x, kH, kW, sH, sW, pH, pW)   # (B, C_in*kH*kW, L)
    W_mat = weight.reshape(C_out, -1)              # (C_out, C_in*kH*kW)
    out   = np.einsum("bkl,ok->bol", col, W_mat)  # (B, C_out, L)
    out   = out.reshape(B, C_out, H_out, W_out)

    if bias is not None:
        out += bias[np.newaxis, :, np.newaxis, np.newaxis]
    return out.astype(np.float32)


def conv_gemm_layer_info(
    name:    str,
    B:       int,
    C_in:    int,
    H:       int, W:       int,
    C_out:   int,
    kH:      int, kW:      int,
    sH:      int = 1, sW: int = 1,
    pH:      int = 0, pW: int = 0,
    groups:  int = 1,
) -> GEMMLayer:
    """Compute GEMM shape and MAC count for a Conv2D layer."""
    H_out = (H + 2 * pH - kH) // sH + 1
    W_out = (W + 2 * pW - kW) // sW + 1
    L     = H_out * W_out
    C_in_g = C_in // groups

    A_shape = (C_out, C_in_g * kH * kW)
    B_shape = (C_in_g * kH * kW, L)
    C_shape = (C_out, L)

    macs = B * C_out * C_in_g * kH * kW * H_out * W_out
    bytes_read  = (A_shape[0] * A_shape[1] + B_shape[0] * B_shape[1]) * 4
    bytes_write = C_out * L * 4

    return GEMMLayer(name, A_shape, B_shape, C_shape, macs, bytes_read, bytes_write)


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: FC layer GEMM
# ─────────────────────────────────────────────────────────────────────────────

def fc_gemm_layer_info(
    name:    str,
    in_dim:  int,
    out_dim: int,
    B:       int = 1,
) -> GEMMLayer:
    """FC layer: W[out_dim, in_dim] @ h[in_dim, B] → out[out_dim, B]."""
    macs        = B * in_dim * out_dim
    bytes_read  = (out_dim * in_dim + in_dim * B) * 4
    bytes_write = out_dim * B * 4
    return GEMMLayer(
        name    = name,
        A_shape = (out_dim, in_dim),
        B_shape = (in_dim, B),
        C_shape = (out_dim, B),
        macs    = macs,
        bytes_read  = bytes_read,
        bytes_write = bytes_write,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full MAC / bandwidth table for DS-CNN Medium
# ─────────────────────────────────────────────────────────────────────────────

def build_mac_table(variant: str = "medium") -> List[GEMMLayer]:
    """
    Return the full list of GEMMLayer entries for DS-CNN (medium variant).

    Based on:
      Input: (1, 1, 40, 101)   [B, C, H, W]
      Stem:  Conv(1→172, 10×4, stride=2×1, pad=4×1) → (1, 172, 21, 101)
             Wait — with H=40, kH=10, sH=2, pH=4:
             H_out = (40 + 8 - 10)//2 + 1 = 38//2+1 = 20
             W_out = (101 + 2 - 4)//1 + 1 = 99+1 = 100
             → (1, 172, 20, 100)
      Body:  4× DW(172,3×3,p=1) + PW(172→172,1×1)
             → spatially same (1, 172, 20, 100)
      GAP:   (1, 172, 1, 1) → flatten (1, 172)
      FC1:   (1, 172) → (1, 172)
      FC2:   (1, 172) → (1, 12)
    """
    configs = {
        "small":  dict(C=64,  n=2,  fc=128),
        "medium": dict(C=172, n=4,  fc=172),
        "large":  dict(C=276, n=5,  fc=276),
    }
    cfg = configs[variant]
    C, n_blocks, fc_dim = cfg["C"], cfg["n"], cfg["fc"]

    # Compute spatial sizes through stem
    H, W = 40, 101
    kH_s, kW_s = 10, 4
    sH_s, sW_s = 2, 1
    pH_s, pW_s = 4, 1
    H_s = (H + 2 * pH_s - kH_s) // sH_s + 1   # 20
    W_s = (W + 2 * pW_s - kW_s) // sW_s + 1   # 100

    layers: List[GEMMLayer] = []

    # Preprocessing GEMM
    layers.append(mel_gemm_layer_info())

    # Stem Conv
    layers.append(conv_gemm_layer_info(
        "Stem Conv(1→C, 10×4, s=2×1)", 1, 1, H, W, C,
        kH_s, kW_s, sH_s, sW_s, pH_s, pW_s,
    ))

    # DW-Sep blocks
    for i in range(n_blocks):
        layers.append(conv_gemm_layer_info(
            f"Block{i+1} DW-Conv(C, 3×3, s=1)",
            1, C, H_s, W_s, C, 3, 3, 1, 1, 1, 1, groups=C,
        ))
        layers.append(conv_gemm_layer_info(
            f"Block{i+1} PW-Conv(C→C, 1×1)",
            1, C, H_s, W_s, C, 1, 1, 1, 1, 0, 0,
        ))

    # FC layers (implemented as Linear in PyTorch, equivalent to GEMM)
    layers.append(fc_gemm_layer_info("FC1", C, fc_dim))
    layers.append(fc_gemm_layer_info("FC2 (classifier)", fc_dim, 12))

    return layers


def print_mac_table(variant: str = "medium") -> None:
    """Print the per-layer MAC / bandwidth table."""
    layers = build_mac_table(variant)
    total_macs  = sum(l.macs for l in layers)
    total_bytes = sum(l.bytes_read + l.bytes_write for l in layers)

    print(f"\n{'='*100}")
    print(f"DS-CNN {variant.upper()} — GEMM layer breakdown")
    print(f"{'='*100}")
    print(f"  {'Layer':<30}  {'A shape':>18}  {'B shape':>18}  "
          f"{'MACs':>12}  {'Bytes rd':>12}  {'AI (M/B)':>10}")
    print(f"  {'-'*96}")
    for l in layers:
        print(f"  {l.name:<30}  {str(l.A_shape):>18}  {str(l.B_shape):>18}  "
              f"{l.macs:>12,}  {l.bytes_read:>12,}  {l.arithmetic_intensity:>10.2f}")
    print(f"  {'─'*96}")
    print(f"  {'TOTAL':<30}  {'':>18}  {'':>18}  "
          f"{total_macs:>12,}  {total_bytes:>12,}")
    print(f"\n  Total MACs: {total_macs/1e6:.2f} M")
    print(f"  Total data: {total_bytes/1024:.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Full NumPy GEMM reference inference (no PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

class GEMMPipeline:
    """
    Complete KWS pipeline as pure NumPy GEMM operations.

    Used to:
      1. Verify the HLS kernel output (Step 13 C-sim testbench)
      2. Confirm all configs produce the same prediction (Step 22)

    Note: This does NOT load real trained weights — it uses random
    weights unless you provide a state dict. It is a correctness reference,
    not a trained model.
    """

    def __init__(self, variant: str = "medium", seed: int = 42):
        from pipeline.model import MODEL_CONFIGS
        rng = np.random.default_rng(seed)
        stem_ch, n_blocks, dw_ch, fc_dim = MODEL_CONFIGS[variant]
        self.variant  = variant
        self.n_blocks = n_blocks
        self.dw_ch    = dw_ch
        self.fc_dim   = fc_dim
        self.M_filt   = build_mel_filterbank()

        # Random weights (replace with loaded weights for real validation)
        scale = 0.01
        self.W_stem = rng.standard_normal((stem_ch, 1, 10, 4)).astype(np.float32) * scale
        self.dw_weights = [
            rng.standard_normal((dw_ch, 1, 3, 3)).astype(np.float32) * scale
            for _ in range(n_blocks)
        ]
        self.pw_weights = [
            rng.standard_normal((dw_ch, dw_ch, 1, 1)).astype(np.float32) * scale
            for _ in range(n_blocks)
        ]
        self.W_fc1   = rng.standard_normal((fc_dim, dw_ch)).astype(np.float32) * scale
        self.b_fc1   = np.zeros(fc_dim, dtype=np.float32)
        self.W_fc2   = rng.standard_normal((12, fc_dim)).astype(np.float32) * scale
        self.b_fc2   = np.zeros(12, dtype=np.float32)

    def preprocess(self, wav: np.ndarray) -> np.ndarray:
        """WAV → log-mel features via NumPy GEMM. Returns (1, 1, 40, 101)."""
        frames    = frame_signal(wav)
        power     = power_spectrum(frames)
        mel_spec  = mel_gemm_numpy(power, self.M_filt)  # GEMM Stage 1
        log_mel   = log_compression(mel_spec)
        # Add batch + channel dims: (1, 1, 40, 101)
        return log_mel[np.newaxis, np.newaxis, :, :]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass via im2col GEMM.
        Args:
            x: (1, 1, 40, 101)
        Returns:
            logits: (12,)
        """
        # Stem Conv (GEMM Stage 2a)
        x = conv_gemm(x, self.W_stem, stride=(2, 1), padding=(4, 1))
        x = np.maximum(x, 0)  # ReLU (no BN weights in reference — omitted)

        # DW-Sep blocks (GEMM Stages 2b...)
        for i in range(self.n_blocks):
            # Depthwise conv (groups = dw_ch)
            dw_out = np.zeros_like(x)
            for g in range(self.dw_ch):
                w_g = self.dw_weights[i][g:g+1, :, :, :]
                dw_out[:, g:g+1] = conv_gemm(
                    x[:, g:g+1], w_g, stride=(1, 1), padding=(1, 1)
                )
            x = np.maximum(dw_out, 0)
            # Pointwise conv (groups = 1)
            x = conv_gemm(x, self.pw_weights[i], stride=(1, 1))
            x = np.maximum(x, 0)

        # Global Average Pool
        x = x.mean(axis=(2, 3))    # (1, dw_ch)

        # FC1 (GEMM Stage 3a)
        x = x @ self.W_fc1.T + self.b_fc1   # (1, fc_dim)
        x = np.maximum(x, 0)

        # FC2 (GEMM Stage 3b)
        x = x @ self.W_fc2.T + self.b_fc2   # (1, 12)
        return x[0]                           # (12,)

    def run(self, wav: np.ndarray) -> np.ndarray:
        """End-to-end: WAV → logits."""
        feat = self.preprocess(wav)
        return self.forward(feat)


if __name__ == "__main__":
    print_mac_table("small")
    print_mac_table("medium")
    print_mac_table("large")

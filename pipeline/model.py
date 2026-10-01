"""
pipeline/model.py
-----------------
Step 5 — DS-CNN model definitions.

Four variants:
  - DSCNN_S  (Small)   ~35K params,  ~5M MACs
  - DSCNN_M  (Medium)  ~423K params, ~74M MACs
  - DSCNN_L  (Large)   ~1.9M params, ~237M MACs
  - DSCNN_GRU          Medium backbone + GRU tail  (deliberately unsupported op
                        to expose DPU CPU-fallback subgraph split)

Input shape: (B, 1, N_MELS, T) = (B, 1, 40, 101)  — NCHW for PyTorch
DPU input:   NHWC  →  transposed in the VART runner

All layers use DPU-friendly ops only (DSCNN_S/M/L):
  Conv2D, DW-Conv2D, PW-Conv2D (1x1), BN, ReLU, GlobalAvgPool, FC (Linear).
Softmax is always on the CPU (not DPU-supported).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from pipeline.utils import N_MELS, NUM_CLASSES, NUM_FRAMES

# ── Model configs ─────────────────────────────────────────────────────────────
#   (stem_channels, n_dw_blocks, dw_channels, fc_dim)
MODEL_CONFIGS: Dict[str, tuple] = {
    "small":  (64,  2,  64,  128),
    "medium": (172, 4, 172,  172),
    "large":  (276, 5, 276,  276),
}


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Standard Conv2D + BatchNorm + ReLU. Fused into single DPU instruction."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size,
        stride=1,
        padding=0,
        groups: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size,
            stride=stride, padding=padding,
            groups=groups, bias=bias,
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    """
    DW-Conv2D(3x3) + BN + ReLU  +  PW-Conv2D(1x1) + BN + ReLU.
    All four ops (DW, BN, ReLU, PW) are DPU-native.
    """

    def __init__(self, channels: int, bias: bool = False):
        super().__init__()
        # Depthwise: groups=channels
        self.dw = ConvBNReLU(channels, channels,
                             kernel_size=3, padding=1, groups=channels, bias=bias)
        # Pointwise: 1×1 conv, groups=1
        self.pw = ConvBNReLU(channels, channels,
                             kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ─────────────────────────────────────────────────────────────────────────────
# DS-CNN backbone (S / M / L)
# ─────────────────────────────────────────────────────────────────────────────

class DSCNN(nn.Module):
    """
    Depthwise Separable CNN for Keyword Spotting.

    Architecture (following "Hello Edge", Zhang et al. 2017):
      Stem: Conv2D(1→C, 10×4, stride 2×1) + BN + ReLU
      Body: N × DepthwiseSeparableBlock(C channels)
      Head: GlobalAvgPool → FC(C→fc_dim) → Dropout → FC(fc_dim→num_classes)

    Note: Softmax is applied OUTSIDE the model (in postprocessing.py)
    so that the exported ONNX / xmodel excludes it from the DPU subgraph.
    """

    def __init__(
        self,
        variant:     str  = "medium",
        num_classes: int  = NUM_CLASSES,
        dropout:     float = 0.5,
        n_mels:      int  = N_MELS,
        n_frames:    int  = NUM_FRAMES,
    ):
        super().__init__()
        stem_ch, n_blocks, dw_ch, fc_dim = MODEL_CONFIGS[variant]
        self.variant = variant

        # Stem convolution
        # Input: (B, 1, 40, 101)  → Output: (B, stem_ch, 16, 101)
        self.stem = ConvBNReLU(
            1, stem_ch,
            kernel_size=(10, 4),
            stride=(2, 1),
            padding=(4, 1),
        )

        # Transition: adapt channels if stem ≠ dw_ch
        if stem_ch != dw_ch:
            self.adapt = ConvBNReLU(stem_ch, dw_ch, kernel_size=1)
        else:
            self.adapt = nn.Identity()

        # DW-Separable blocks
        self.body = nn.Sequential(
            *[DepthwiseSeparableBlock(dw_ch) for _ in range(n_blocks)]
        )

        # Head
        self.gap     = nn.AdaptiveAvgPool2d(1)            # → (B, dw_ch, 1, 1)
        self.flatten = nn.Flatten()                        # → (B, dw_ch)
        self.fc1     = nn.Linear(dw_ch, fc_dim, bias=True)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2     = nn.Linear(fc_dim, num_classes, bias=True)
        # NOTE: NO Softmax here — applied in postprocessing.py

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N_MELS, T) = (B, 1, 40, 101)
        Returns:
            logits: (B, num_classes)
        """
        x = self.stem(x)        # (B, stem_ch, 16, 101)
        x = self.adapt(x)       # (B, dw_ch,   16, 101)
        x = self.body(x)        # (B, dw_ch,   16, 101)
        x = self.gap(x)         # (B, dw_ch,   1,  1)
        x = self.flatten(x)     # (B, dw_ch)
        x = self.relu_fc(self.fc1(x))   # (B, fc_dim)
        x = self.dropout(x)
        return self.fc2(x)      # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Unsupported-op variant — deliberately triggers DPU CPU-fallback
# ─────────────────────────────────────────────────────────────────────────────

class DSCNN_GRU(nn.Module):
    """
    DS-CNN with a GRU-based temporal head instead of GAP → FC.

    The GRU layer is NOT supported by DPUCZDX8G, so the Vitis AI compiler
    will split the graph into:
      Subgraph 0 (DPU): stem + body
      Subgraph 1 (CPU): GRU + FC
    This documents exactly where the partition happens and what the
    CPU-fallback overhead is.

    Used in Step 10 (operator support map) and Step 17 (Config B unsupported-op).
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dw_ch:       int = 172,
        n_blocks:    int = 4,
        gru_hidden:  int = 128,
        dropout:     float = 0.5,
    ):
        super().__init__()
        # Same stem + body as DSCNN Medium
        stem_ch = 172
        self.stem = ConvBNReLU(
            1, stem_ch, kernel_size=(10, 4), stride=(2, 1), padding=(4, 1)
        )
        self.body = nn.Sequential(
            *[DepthwiseSeparableBlock(dw_ch) for _ in range(n_blocks)]
        )
        # GRU head — NOT DPU-supported  ← graph split here
        self.gru     = nn.GRU(
            input_size=dw_ch, hidden_size=gru_hidden,
            num_layers=1, batch_first=True
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(gru_hidden, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, N_MELS, T) = (B, 1, 40, 101)
        Returns:
            logits: (B, num_classes)
        """
        B = x.size(0)
        x = self.stem(x)                # (B, 172, 16, 101)
        x = self.body(x)                # (B, 172, 16, 101)
        # Pool over frequency, keep time: (B, 172, T') → (B, T', 172)
        x = x.mean(dim=2)              # (B, 172, 101)
        x = x.permute(0, 2, 1)        # (B, 101, 172)
        _, h_n = self.gru(x)           # h_n: (1, B, gru_hidden)
        x = h_n.squeeze(0)             # (B, gru_hidden) — ← DPU→CPU boundary
        x = self.dropout(x)
        return self.fc(x)              # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_model(variant: str = "medium", **kwargs) -> nn.Module:
    """
    Factory function.

    Args:
        variant: "small" | "medium" | "large" | "gru"
    Returns:
        nn.Module
    """
    if variant == "gru":
        return DSCNN_GRU(**kwargs)
    if variant not in MODEL_CONFIGS:
        raise ValueError(f"Unknown variant '{variant}'. "
                         f"Choose from {list(MODEL_CONFIGS) + ['gru']}")
    return DSCNN(variant=variant, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# MAC / parameter counting
# ─────────────────────────────────────────────────────────────────────────────

def count_params(model: nn.Module) -> int:
    """Total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_macs(model: nn.Module, input_shape=(1, 1, N_MELS, NUM_FRAMES)) -> int:
    """
    Estimate MACs using a forward hook on Conv2d and Linear layers.

    Returns approximate total MACs (multiply-accumulate ops).
    """
    total_macs = 0

    def conv_hook(module, inp, out):
        nonlocal total_macs
        B, C_out, H_out, W_out = out.shape
        C_in = inp[0].shape[1]
        kH, kW = module.kernel_size
        groups = module.groups
        # MACs = C_out * H_out * W_out * (C_in/groups) * kH * kW
        total_macs += B * C_out * H_out * W_out * (C_in // groups) * kH * kW

    def linear_hook(module, inp, out):
        nonlocal total_macs
        B = inp[0].shape[0]
        total_macs += B * module.in_features * module.out_features

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(*input_shape, device=next(model.parameters()).device)
        model(dummy)

    for h in hooks:
        h.remove()
    return total_macs


def model_summary(variant: str = "medium") -> None:
    """Print parameter count and MACs for a model variant."""
    model = build_model(variant)
    params = count_params(model)
    macs   = count_macs(model)
    print(f"\nDS-CNN {variant.upper()}")
    print(f"  Parameters : {params:,}")
    print(f"  MACs/infer : {macs:,.0f}  ({macs/1e6:.1f} M)")


if __name__ == "__main__":
    for v in ["small", "medium", "large", "gru"]:
        try:
            model_summary(v)
        except Exception as e:
            print(f"  {v}: {e}")

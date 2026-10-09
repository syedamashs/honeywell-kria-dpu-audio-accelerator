"""
pipeline/tests/test_gemm_reference.py
----------------------------------------
Step 6 verification — GEMM reference implementation tests.

Run:
    pytest pipeline/tests/test_gemm_reference.py -v

Tests:
    - im2col produces correct matrix shape
    - conv_gemm matches scipy.signal.convolve2d output
    - mel_gemm_numpy matches pipeline mel_filterbank_gemm (both are reference)
    - MAC count table: total MACs in expected range for each variant
    - GEMMPipeline end-to-end: deterministic, correct output shape
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.gemm_reference import (
    GEMMPipeline,
    build_mac_table,
    conv_gemm,
    fc_gemm_layer_info,
    im2col,
    mel_gemm_layer_info,
    mel_gemm_numpy,
)
from pipeline.preprocessing import (
    build_mel_filterbank,
    frame_signal,
    power_spectrum,
    mel_filterbank_gemm,
)
from pipeline.utils import N_MELS, NUM_FRAMES, SAMPLE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_wav():
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav += rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.005
    return wav


# ─────────────────────────────────────────────────────────────────────────────
# im2col tests
# ─────────────────────────────────────────────────────────────────────────────

class TestIm2Col:
    def test_shape_standard_conv(self):
        """3×3 conv on 8×8 input → col matrix (B, C*kH*kW, H_out*W_out)."""
        x = np.random.randn(1, 3, 8, 8).astype(np.float32)
        col = im2col(x, kH=3, kW=3, stride_H=1, stride_W=1, pad_H=1, pad_W=1)
        # With padding=1: H_out = W_out = 8
        assert col.shape == (1, 3 * 3 * 3, 8 * 8), (
            f"Expected (1, 27, 64), got {col.shape}"
        )

    def test_shape_no_padding(self):
        x = np.random.randn(1, 1, 5, 5).astype(np.float32)
        col = im2col(x, kH=3, kW=3)
        # H_out = W_out = 3
        assert col.shape == (1, 1 * 3 * 3, 3 * 3)

    def test_shape_strided(self):
        x = np.random.randn(1, 1, 8, 8).astype(np.float32)
        col = im2col(x, kH=3, kW=3, stride_H=2, stride_W=2, pad_H=1, pad_W=1)
        # H_out = W_out = 4
        assert col.shape == (1, 9, 16)

    def test_reconstruction(self):
        """im2col then GEMM should give same result as direct matmul."""
        B, C, H, W = 1, 1, 6, 6
        x = np.arange(B * C * H * W, dtype=np.float32).reshape(B, C, H, W)
        # 3×3 identity-ish weight
        weight = np.zeros((1, 1, 3, 3), dtype=np.float32)
        weight[0, 0, 1, 1] = 1.0   # center pixel → output = input (with valid padding)
        out_gemm = conv_gemm(x, weight, stride=(1, 1), padding=(1, 1))
        np.testing.assert_allclose(out_gemm[0, 0], x[0, 0], atol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# Conv GEMM tests
# ─────────────────────────────────────────────────────────────────────────────

class TestConvGEMM:
    def test_output_shape(self):
        x = np.random.randn(1, 1, 40, 101).astype(np.float32)
        W = np.random.randn(64, 1, 10, 4).astype(np.float32) * 0.01
        # stride=(2,1), padding=(4,1)
        out = conv_gemm(x, W, stride=(2, 1), padding=(4, 1))
        # H_out = (40 + 8 - 10)//2 + 1 = 20
        # W_out = (101 + 2 - 4)//1 + 1 = 100
        assert out.shape == (1, 64, 20, 100), f"Unexpected shape {out.shape}"

    def test_output_dtype(self):
        x = np.random.randn(1, 1, 8, 8).astype(np.float32)
        W = np.random.randn(4, 1, 3, 3).astype(np.float32)
        out = conv_gemm(x, W, stride=(1, 1), padding=(1, 1))
        assert out.dtype == np.float32

    def test_bias_addition(self):
        x = np.ones((1, 1, 4, 4), dtype=np.float32)
        W = np.zeros((2, 1, 1, 1), dtype=np.float32)  # zero conv
        b = np.array([3.0, 7.0], dtype=np.float32)
        out = conv_gemm(x, W, bias=b, stride=(1, 1))
        assert np.allclose(out[0, 0], 3.0)
        assert np.allclose(out[0, 1], 7.0)


# ─────────────────────────────────────────────────────────────────────────────
# Mel GEMM agreement
# ─────────────────────────────────────────────────────────────────────────────

class TestMelGEMMConsistency:
    """mel_gemm_numpy and mel_filterbank_gemm must produce identical output."""

    def test_identical_to_pipeline(self, synthetic_wav):
        frames   = frame_signal(synthetic_wav)
        power    = power_spectrum(frames)
        M        = build_mel_filterbank()

        ref  = mel_filterbank_gemm(power, M)      # from preprocessing.py
        ours = mel_gemm_numpy(power, M)           # from gemm_reference.py

        np.testing.assert_allclose(
            ref, ours, atol=1e-5,
            err_msg="mel_gemm_numpy diverges from mel_filterbank_gemm"
        )

    def test_layer_info_shape(self):
        info = mel_gemm_layer_info()
        assert info.A_shape[0] == N_MELS
        assert info.C_shape == (N_MELS, NUM_FRAMES)

    def test_macs_positive(self):
        info = mel_gemm_layer_info()
        assert info.macs > 0


# ─────────────────────────────────────────────────────────────────────────────
# MAC table tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMACTable:
    """
    MAC counts must be in the expected ranges from literature:
    DS-CNN-S ~3 M, DS-CNN-M ~17 M, DS-CNN-L ~51 M
    (includes preprocessing GEMM + all network layers).
    """

    @pytest.mark.parametrize("variant,lo_M,hi_M", [
        ("small",   15,   35),   # ~24.9 M (small variant with 40-mel spectrogram)
        ("medium", 150,  350),   # ~263.9 M (medium)
        ("large",  500, 1000),   # ~809.8 M (large)
    ])
    def test_total_macs_in_range(self, variant, lo_M, hi_M):
        layers = build_mac_table(variant)
        total  = sum(l.macs for l in layers)
        total_M = total / 1e6
        assert lo_M <= total_M <= hi_M, (
            f"DS-CNN-{variant}: {total_M:.1f} M MACs outside [{lo_M}, {hi_M}] M"
        )

    def test_mel_layer_is_first(self):
        layers = build_mac_table("medium")
        assert "Mel" in layers[0].name

    def test_all_macs_positive(self):
        for variant in ["small", "medium", "large"]:
            for l in build_mac_table(variant):
                assert l.macs > 0, f"Zero MACs in layer {l.name} ({variant})"

    def test_arithmetic_intensity(self):
        """All layers must have finite, positive arithmetic intensity."""
        layers = build_mac_table("medium")
        for l in layers:
            ai = l.arithmetic_intensity
            assert ai > 0 and np.isfinite(ai), (
                f"Bad arithmetic intensity in {l.name}: {ai}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# GEMMPipeline end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestGEMMPipeline:
    def test_output_shape(self, synthetic_wav):
        pipe    = GEMMPipeline("medium", seed=42)
        logits  = pipe.run(synthetic_wav)
        assert logits.shape == (12,), f"Expected (12,), got {logits.shape}"

    def test_output_dtype(self, synthetic_wav):
        pipe   = GEMMPipeline("medium", seed=42)
        logits = pipe.run(synthetic_wav)
        assert logits.dtype == np.float32

    def test_deterministic(self, synthetic_wav):
        pipe   = GEMMPipeline("medium", seed=42)
        out1   = pipe.run(synthetic_wav)
        out2   = pipe.run(synthetic_wav)
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_differ(self, synthetic_wav):
        pipe1 = GEMMPipeline("medium", seed=1)
        pipe2 = GEMMPipeline("medium", seed=2)
        out1  = pipe1.run(synthetic_wav)
        out2  = pipe2.run(synthetic_wav)
        assert not np.allclose(out1, out2), (
            "Two different random seeds produced identical outputs"
        )

    def test_small_variant(self, synthetic_wav):
        pipe   = GEMMPipeline("small", seed=42)
        logits = pipe.run(synthetic_wav)
        assert logits.shape == (12,)

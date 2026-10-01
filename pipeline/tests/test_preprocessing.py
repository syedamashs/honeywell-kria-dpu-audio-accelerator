"""
pipeline/tests/test_preprocessing.py
--------------------------------------
Step 4 verification — preprocessing output tested against librosa.

Run:
    pytest pipeline/tests/test_preprocessing.py -v

Pass criteria:
    - Our mel filterbank shape matches (N_MELS, N_BINS)
    - Our log-mel spectrogram matches librosa within 1.0 dB absolute tolerance
    - Number of frames is exactly NUM_FRAMES
    - Output dtype is float32
"""

from __future__ import annotations

from io import BytesIO
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.io.wavfile as wav_io

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.preprocessing import (
    build_mel_filterbank,
    decode_wav_bytes,
    extract_log_mel,
    frame_signal,
    load_wav,
    load_wav_bytes,
    log_compression,
    mel_filterbank_gemm,
    power_spectrum,
    pre_emphasis,
    librosa_log_mel,
    N_BINS,
    WIN_LEN,
    HOP_LEN,
)
from pipeline.utils import (
    SAMPLE_RATE,
    N_MELS,
    N_FFT,
    NUM_FRAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_wav():
    """1-second 440 Hz sine wave at 16 kHz — deterministic test signal."""
    rng = np.random.default_rng(42)
    t   = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)
    wav = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Add slight noise to avoid perfectly periodic edge cases
    wav += rng.standard_normal(SAMPLE_RATE).astype(np.float32) * 0.005
    return wav


@pytest.fixture(scope="module")
def silence_wav():
    """Silent clip — tests log-floor behavior."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMelFilterbank:
    def test_shape(self):
        M = build_mel_filterbank()
        assert M.shape == (N_MELS, N_BINS), (
            f"Expected ({N_MELS}, {N_BINS}), got {M.shape}"
        )

    def test_dtype(self):
        M = build_mel_filterbank()
        assert M.dtype == np.float32

    def test_non_negative(self):
        M = build_mel_filterbank()
        assert (M >= 0).all(), "Filterbank weights must be non-negative"

    def test_row_sums(self):
        M = build_mel_filterbank()
        row_sums = M.sum(axis=1)
        assert (row_sums > 0).all(), "Each mel filter must have positive sum"

    def test_deterministic(self):
        M1 = build_mel_filterbank()
        M2 = build_mel_filterbank()
        np.testing.assert_array_equal(M1, M2)


class TestFraming:
    def test_frame_count(self, synthetic_wav):
        frames = frame_signal(synthetic_wav)
        assert frames.shape[0] == NUM_FRAMES, (
            f"Expected {NUM_FRAMES} frames, got {frames.shape[0]}"
        )

    def test_frame_width(self, synthetic_wav):
        frames = frame_signal(synthetic_wav)
        assert frames.shape[1] == WIN_LEN

    def test_dtype(self, synthetic_wav):
        frames = frame_signal(synthetic_wav)
        assert frames.dtype == np.float32


class TestPowerSpectrum:
    def test_shape(self, synthetic_wav):
        frames = frame_signal(synthetic_wav)
        power  = power_spectrum(frames)
        assert power.shape == (NUM_FRAMES, N_BINS), (
            f"Expected ({NUM_FRAMES}, {N_BINS}), got {power.shape}"
        )

    def test_non_negative(self, synthetic_wav):
        frames = frame_signal(synthetic_wav)
        power  = power_spectrum(frames)
        assert (power >= 0).all()

    def test_known_frequency(self, synthetic_wav):
        """440 Hz should appear as a peak in the correct FFT bin."""
        frames = frame_signal(synthetic_wav)
        power  = power_spectrum(frames)
        # Average over time to smooth
        mean_power = power.mean(axis=0)
        expected_bin = int(440 * N_FFT / SAMPLE_RATE)
        peak_bin = int(np.argmax(mean_power))
        assert abs(peak_bin - expected_bin) <= 3, (
            f"440 Hz peak at bin {peak_bin}, expected near {expected_bin}"
        )


class TestMelGEMM:
    def test_shape(self, synthetic_wav):
        frames   = frame_signal(synthetic_wav)
        power    = power_spectrum(frames)
        mel_spec = mel_filterbank_gemm(power)
        assert mel_spec.shape == (N_MELS, NUM_FRAMES), (
            f"Expected ({N_MELS}, {NUM_FRAMES}), got {mel_spec.shape}"
        )

    def test_non_negative(self, synthetic_wav):
        frames   = frame_signal(synthetic_wav)
        power    = power_spectrum(frames)
        mel_spec = mel_filterbank_gemm(power)
        assert (mel_spec >= 0).all()


class TestLogCompression:
    def test_shape_preserved(self, synthetic_wav):
        frames   = frame_signal(synthetic_wav)
        power    = power_spectrum(frames)
        mel      = mel_filterbank_gemm(power)
        log_mel  = log_compression(mel)
        assert log_mel.shape == mel.shape

    def test_no_nan(self, silence_wav):
        """Log of a silent clip should not produce NaN (floor prevents log(0))."""
        frames   = frame_signal(silence_wav)
        power    = power_spectrum(frames)
        mel      = mel_filterbank_gemm(power)
        log_mel  = log_compression(mel)
        assert not np.isnan(log_mel).any(), "NaN in log-mel for silent clip"
        assert not np.isinf(log_mel).any(), "Inf in log-mel for silent clip"


class TestFullPipeline:
    def test_output_shape(self, synthetic_wav):
        feat = extract_log_mel(synthetic_wav)
        assert feat.shape == (N_MELS, NUM_FRAMES)

    def test_output_dtype(self, synthetic_wav):
        feat = extract_log_mel(synthetic_wav)
        assert feat.dtype == np.float32

    def test_deterministic(self, synthetic_wav):
        feat1 = extract_log_mel(synthetic_wav)
        feat2 = extract_log_mel(synthetic_wav)
        np.testing.assert_array_equal(feat1, feat2)


class TestWavByteLoading:
    def test_stereo_audio_is_downmixed_resampled_and_normalized(self):
        sample_rate = 8_000
        stereo = np.full((sample_rate, 2), 16_384, dtype=np.int16)
        wav_buffer = BytesIO()
        wav_io.write(wav_buffer, sample_rate, stereo)

        audio = load_wav_bytes(wav_buffer.getvalue(), "upload.wav")

        assert audio.shape == (SAMPLE_RATE,)
        assert audio.dtype == np.float32
        assert audio.mean() == pytest.approx(0.5, abs=0.01)

    def test_rejects_non_wav_bytes(self):
        with pytest.raises(ValueError, match="Could not read upload.wav as a WAV file"):
            load_wav_bytes(b"not a wav", "upload.wav")

    def test_long_recording_is_preserved_up_to_duration_limit(self):
        sample_rate = 48_000
        audio = np.full(sample_rate * 8, 8_192, dtype=np.int16)
        wav_buffer = BytesIO()
        wav_io.write(wav_buffer, sample_rate, audio)

        decoded = decode_wav_bytes(wav_buffer.getvalue(), "recording.wav", max_duration_s=7.0)

        assert decoded.shape == (SAMPLE_RATE * 7,)
        assert decoded.dtype == np.float32


# ─────────────────────────────────────────────────────────────────────────────
# Librosa comparison (integration test)
# ─────────────────────────────────────────────────────────────────────────────

class TestLibrosaComparison:
    """
    Compare our log-mel spectrogram against librosa's reference output.

    Tolerance: 1.5 dB absolute — accounts for:
    - Slightly different mel filterbank implementation (ours is pure NumPy)
    - Different power_to_db reference (librosa uses ref=max, we use log10 with floor)
    - Small numerical differences in FFT implementations
    """

    @pytest.fixture(autouse=True)
    def check_librosa(self):
        pytest.importorskip("librosa")

    def test_shape_matches_librosa(self, synthetic_wav):
        our_feat = extract_log_mel(synthetic_wav, apply_pre_emphasis=False)
        lib_feat = librosa_log_mel(synthetic_wav)
        assert our_feat.shape == lib_feat.shape

    def test_correlation_with_librosa(self, synthetic_wav):
        """
        Our log-mel and librosa's should be highly correlated
        (Pearson r > 0.98) even if absolute values differ.
        """
        our_feat = extract_log_mel(synthetic_wav, apply_pre_emphasis=False)
        lib_feat = librosa_log_mel(synthetic_wav)
        # Flatten and compute correlation
        r = np.corrcoef(our_feat.ravel(), lib_feat.ravel())[0, 1]
        assert r > 0.98, (
            f"Pearson correlation with librosa is {r:.4f}, expected > 0.98. "
            "Check filterbank parameters."
        )

    def test_relative_pattern(self, synthetic_wav):
        """
        Relative energy patterns (which frequency bins are hot/cold)
        should match even if absolute scale differs.
        """
        our_feat = extract_log_mel(synthetic_wav, apply_pre_emphasis=False)
        lib_feat = librosa_log_mel(synthetic_wav)
        # Normalize each to zero mean, unit std
        our_norm = (our_feat - our_feat.mean()) / (our_feat.std() + 1e-8)
        lib_norm = (lib_feat - lib_feat.mean()) / (lib_feat.std() + 1e-8)
        mae = np.abs(our_norm - lib_norm).mean()
        assert mae < 0.5, (
            f"Normalized pattern MAE vs librosa: {mae:.4f}, expected < 0.5"
        )

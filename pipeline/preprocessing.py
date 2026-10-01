"""
pipeline/preprocessing.py
--------------------------
Step 4 — Audio preprocessing pipeline.

Stages (all NumPy/SciPy — no librosa dependency at inference time):
  1. Load WAV (16kHz mono, pad/trim to 1s)
  2. Pre-emphasis filter
  3. Framing + Hann windowing
  4. FFT → power spectrum
  5. Mel filterbank applied as GEMM   ← custom HLS kernel target
  6. Log compression (log10 with floor)
  7. Optional DCT → MFCC coefficients

All parameters are imported from utils.py so every config uses identical values.

Verification: output is tested against librosa in pipeline/tests/test_preprocessing.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import scipy.fft as sp_fft
import scipy.io.wavfile as wav_io
import scipy.signal as sp_signal

from pipeline.utils import (
    CLIP_DURATION,
    FMAX,
    FMIN,
    HOP_MS,
    N_FFT,
    N_MELS,
    N_MFCC,
    NUM_FRAMES,
    SAMPLE_RATE,
    WINDOW_MS,
    pad_or_trim,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants derived from utils parameters
# ─────────────────────────────────────────────────────────────────────────────
WIN_LEN  = int(SAMPLE_RATE * WINDOW_MS / 1000)   # 400 samples
HOP_LEN  = int(SAMPLE_RATE * HOP_MS    / 1000)   # 160 samples
N_BINS   = N_FFT // 2 + 1                         # 257 one-sided FFT bins
LOG_FLOOR = 1e-10                                  # prevents log(0)
PRE_EMPH  = 0.97                                   # pre-emphasis coefficient


# ─────────────────────────────────────────────────────────────────────────────
# 1. WAV loading
# ─────────────────────────────────────────────────────────────────────────────

def load_wav(path: str | Path) -> np.ndarray:
    """
    Load a WAV file and return a float32 mono array at SAMPLE_RATE.

    Returns:
        wav: float32 array of shape (SAMPLE_RATE,) = (16000,)
    """
    sr, data = wav_io.read(str(path))
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {path}")
    if data.ndim == 2:                     # stereo → mono
        data = data.mean(axis=1)
    # Normalise to [-1, 1]
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    else:
        data = data.astype(np.float32)
    return pad_or_trim(data, int(SAMPLE_RATE * CLIP_DURATION))


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pre-emphasis
# ─────────────────────────────────────────────────────────────────────────────

def pre_emphasis(wav: np.ndarray, coeff: float = PRE_EMPH) -> np.ndarray:
    """High-pass pre-emphasis filter: y[n] = x[n] - coeff * x[n-1]."""
    return np.append(wav[0], wav[1:] - coeff * wav[:-1]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Framing + windowing
# ─────────────────────────────────────────────────────────────────────────────

def frame_signal(
    wav: np.ndarray,
    win_len: int = WIN_LEN,
    hop_len: int = HOP_LEN,
) -> np.ndarray:
    """
    Split signal into overlapping frames with Hann window applied.

    Returns:
        frames: float32 array of shape (T, win_len)
                where T = NUM_FRAMES = 101
    """
    n_frames = 1 + (len(wav) - win_len) // hop_len
    # Build index matrix for efficient gather
    indices = (
        np.arange(win_len)[np.newaxis, :]          # (1, win_len)
        + np.arange(n_frames)[:, np.newaxis] * hop_len  # (T, 1)
    )
    frames = wav[indices].astype(np.float32)        # (T, win_len)
    window = np.hanning(win_len).astype(np.float32)
    return frames * window                           # (T, win_len)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FFT → power spectrum
# ─────────────────────────────────────────────────────────────────────────────

def power_spectrum(frames: np.ndarray, n_fft: int = N_FFT) -> np.ndarray:
    """
    Compute one-sided power spectrum via FFT.

    Args:
        frames: (T, win_len)
    Returns:
        power: (T, N_BINS) = (T, 257)  — float32
    """
    spec = sp_fft.rfft(frames, n=n_fft, axis=-1)   # (T, N_BINS) complex
    power = (np.abs(spec) ** 2).astype(np.float32)  # (T, N_BINS) real
    return power


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mel filterbank matrix (cached at module load)
# ─────────────────────────────────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(
    n_mels:  int   = N_MELS,
    n_fft:   int   = N_FFT,
    sr:      int   = SAMPLE_RATE,
    fmin:    float = FMIN,
    fmax:    float = FMAX,
) -> np.ndarray:
    """
    Build the mel filterbank matrix  M  of shape (n_mels, N_BINS).

    The mel filterbank GEMM is:
        Mel[n_mels, T] = M[n_mels, N_BINS]  @  P[N_BINS, T]

    This matrix is fixed for the lifetime of the project.
    It is also the matrix stored in BRAM inside the HLS kernel.

    Returns:
        M: float32 array of shape (N_MELS, N_BINS) = (40, 257)
    """
    n_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points  = np.array([_mel_to_hz(m) for m in mel_points])
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    M = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, ctr):
            M[m - 1, k] = (k - lo) / max(ctr - lo, 1)
        for k in range(ctr, hi):
            M[m - 1, k] = (hi - k) / max(hi - ctr, 1)
    return M


# Module-level cached filterbank — built once, reused for every call
_MEL_FILTERBANK: Optional[np.ndarray] = None


def get_mel_filterbank() -> np.ndarray:
    """Return the cached (40, 257) mel filterbank matrix."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is None:
        _MEL_FILTERBANK = build_mel_filterbank()
    return _MEL_FILTERBANK


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Mel GEMM  (this is the operation offloaded to the HLS kernel in Config C)
# ─────────────────────────────────────────────────────────────────────────────

def mel_filterbank_gemm(
    power: np.ndarray,
    M: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Apply mel filterbank via matrix multiplication (GEMM).

    Math:
        Mel[n_mels, T] = M[n_mels, N_BINS]  @  P.T[N_BINS, T]

    Args:
        power: (T, N_BINS) float32 power spectrum
        M:     (N_MELS, N_BINS) filterbank matrix — uses cached if None
    Returns:
        mel_spec: (n_mels, T) float32   shape (40, 101)
    """
    if M is None:
        M = get_mel_filterbank()
    # P is (T, N_BINS); M @ P.T gives (N_MELS, T)
    mel_spec = M @ power.T                          # (40, 257) @ (257, 101) → (40, 101)
    return mel_spec.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Log compression
# ─────────────────────────────────────────────────────────────────────────────

def log_compression(mel_spec: np.ndarray, floor: float = LOG_FLOOR) -> np.ndarray:
    """
    Apply log10 compression with a floor to prevent log(0).

    Args:
        mel_spec: (n_mels, T) float32
    Returns:
        log_mel: (n_mels, T) float32
    """
    return np.log10(np.maximum(mel_spec, floor)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Optional DCT → MFCC
# ─────────────────────────────────────────────────────────────────────────────

def dct_mfcc(log_mel: np.ndarray, n_mfcc: int = N_MFCC) -> np.ndarray:
    """
    Apply Type-II DCT along the frequency axis to produce MFCC.

    Args:
        log_mel: (n_mels, T) float32
    Returns:
        mfcc: (n_mfcc, T) float32
    """
    mfcc_all = sp_signal.cosine_transform(log_mel, type=2, norm="ortho", axis=0)
    return mfcc_all[:n_mfcc].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def extract_log_mel(
    wav: np.ndarray,
    apply_pre_emphasis: bool = True,
    return_mfcc: bool = False,
) -> np.ndarray:
    """
    Full preprocessing pipeline: WAV array → log-mel spectrogram (or MFCC).

    Args:
        wav:               float32 array of shape (16000,)
        apply_pre_emphasis: whether to apply high-pass filter
        return_mfcc:       if True, apply DCT and return MFCC instead
    Returns:
        features: float32 array of shape (N_MELS, T) = (40, 101)
                  or (N_MFCC, T) if return_mfcc=True
    """
    if apply_pre_emphasis:
        wav = pre_emphasis(wav)
    frames   = frame_signal(wav)                  # (T, 400)
    power    = power_spectrum(frames)             # (T, 257)
    mel_spec = mel_filterbank_gemm(power)         # (40, 101)  ← GEMM
    log_mel  = log_compression(mel_spec)          # (40, 101)
    if return_mfcc:
        return dct_mfcc(log_mel)                  # (40, 101)
    return log_mel                                # (40, 101)


def extract_log_mel_from_file(path: str | Path, **kwargs) -> np.ndarray:
    """Convenience: load WAV file and return log-mel features."""
    return extract_log_mel(load_wav(path), **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Librosa verification helper
# ─────────────────────────────────────────────────────────────────────────────

def librosa_log_mel(wav: np.ndarray) -> np.ndarray:
    """
    Compute log-mel spectrogram using librosa (ground-truth reference).
    Used only in tests — not at inference time.

    Returns:
        log_mel: (N_MELS, T) float32
    """
    import librosa  # deferred import: not needed at board runtime

    S = librosa.feature.melspectrogram(
        y=wav,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN,
        window="hann",
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        center=False,
    )
    return librosa.power_to_db(S, ref=1.0).astype(np.float32)

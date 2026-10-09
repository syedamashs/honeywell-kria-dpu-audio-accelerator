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

from io import BytesIO
from math import gcd
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:
    import scipy.fft as sp_fft
except ImportError:
    sp_fft = None

try:
    import scipy.io.wavfile as wav_io
except ImportError:
    wav_io = None

try:
    import scipy.signal as sp_signal
except ImportError:
    sp_signal = None


from pipeline.utils import (
    CLIP_DURATION,
    FMAX,
    FMIN,
    HOP_LEN,
    HOP_MS,
    N_BINS,
    N_FFT,
    N_MELS,
    N_MFCC,
    NUM_FRAMES,
    SAMPLE_RATE,
    WINDOW_MS,
    WIN_LEN,
    pad_or_trim,
)

# ─────────────────────────────────────────────────────────────────────────────
# Local constants (not in utils — preprocessing-specific)
# ─────────────────────────────────────────────────────────────────────────────
LOG_FLOOR = 1e-10                                  # prevents log(0)
PRE_EMPH  = 0.97                                   # pre-emphasis coefficient


# ─────────────────────────────────────────────────────────────────────────────
# 1. WAV loading
# ─────────────────────────────────────────────────────────────────────────────

def _read_wav(source: str | Path | BytesIO) -> Tuple[int, np.ndarray]:
    """Read WAV file or bytes using scipy if available, with stdlib wave fallback."""
    if wav_io is not None:
        try:
            return wav_io.read(source)
        except Exception:
            pass

    import wave
    if hasattr(source, "read") or isinstance(source, BytesIO):
        wf = wave.open(source, "rb")
    else:
        wf = wave.open(str(source), "rb")

    with wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        width = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())
        dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
        if width not in dtype_map:
            raise ValueError(f"Unsupported sample width: {width}")
        data = np.frombuffer(frames, dtype=dtype_map[width])
        if ch > 1:
            data = data.reshape(-1, ch)
        return sr, data


def load_wav(path: str | Path) -> np.ndarray:
    """
    Load a WAV file and return a float32 mono array at SAMPLE_RATE.

    Returns:
        wav: float32 array of shape (SAMPLE_RATE,) = (16000,)
    """
    sr, data = _read_wav(path)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} Hz: {path}")
    data = _audio_to_float32_mono(data)
    return pad_or_trim(data, int(SAMPLE_RATE * CLIP_DURATION))


def load_wav_bytes(contents: bytes, source_name: str = "uploaded WAV") -> np.ndarray:
    """Decode uploaded PCM WAV bytes into a normalized 16 kHz mono clip."""
    audio = decode_wav_bytes(contents, source_name)
    return pad_or_trim(audio, int(SAMPLE_RATE * CLIP_DURATION))


def decode_wav_bytes(
    contents: bytes,
    source_name: str = "uploaded WAV",
    max_duration_s: float | None = None,
) -> np.ndarray:
    """Decode WAV bytes to normalized 16 kHz mono audio, preserving duration."""
    try:
        sample_rate, data = _read_wav(BytesIO(contents))
    except Exception as exc:
        raise ValueError(f"Could not read {source_name} as a WAV file: {exc}") from exc

    if not 1_000 <= sample_rate <= 192_000:
        raise ValueError(f"Unsupported sample rate ({sample_rate} Hz) in {source_name}.")

    audio = _audio_to_float32_mono(data)
    if sample_rate != SAMPLE_RATE:
        if sp_signal is not None:
            divisor = gcd(sample_rate, SAMPLE_RATE)
            audio = sp_signal.resample_poly(
                audio,
                up=SAMPLE_RATE // divisor,
                down=sample_rate // divisor,
            ).astype(np.float32)
        else:
            orig_len = len(audio)
            target_len = int(orig_len * SAMPLE_RATE / sample_rate)
            x_old = np.linspace(0, 1, orig_len, endpoint=False)
            x_new = np.linspace(0, 1, target_len, endpoint=False)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)

    if max_duration_s is not None:
        audio = audio[:int(SAMPLE_RATE * max_duration_s)]
    return audio


def _audio_to_float32_mono(data: np.ndarray) -> np.ndarray:
    if data.ndim not in (1, 2):
        raise ValueError(f"Expected mono or stereo audio, got shape {data.shape}.")

    if np.issubdtype(data.dtype, np.integer):
        limits = np.iinfo(data.dtype)
        if limits.min == 0:
            midpoint = (limits.max + 1) / 2.0
            audio = (data.astype(np.float32) - midpoint) / midpoint
        else:
            scale = float(max(abs(limits.min), abs(limits.max)))
            audio = data.astype(np.float32) / scale
    else:
        audio = data.astype(np.float32)

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32)


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
    if sp_fft is not None:
        spec = sp_fft.rfft(frames, n=n_fft, axis=-1)   # (T, N_BINS) complex
    else:
        spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    power = (np.abs(spec) ** 2).astype(np.float32)  # (T, N_BINS) real
    return power


# ─────────────────────────────────────────────────────────────────────────────
# 5. Mel filterbank matrix (cached at module load)
# ─────────────────────────────────────────────────────────────────────────────

def _hz_to_mel(frequencies: np.ndarray | float) -> np.ndarray:
    """Convert Hz to Mel using Slaney Auditory Toolbox formula."""
    frequencies = np.asanyarray(frequencies)
    f_min = 0.0
    f_sp = 200.0 / 3
    mels = (frequencies - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        log_t = frequencies >= min_log_hz
        mels = mels.copy()
        mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels: np.ndarray | float) -> np.ndarray:
    """Convert Mel to Hz using Slaney Auditory Toolbox formula."""
    mels = np.asanyarray(mels)
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        log_t = mels >= min_log_mel
        freqs = freqs.copy()
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        freqs = min_log_hz * np.exp(logstep * (mels - min_log_mel))
    return freqs


def build_mel_filterbank(
    n_mels:  int   = N_MELS,
    n_fft:   int   = N_FFT,
    sr:      int   = SAMPLE_RATE,
    fmin:    float = FMIN,
    fmax:    float = FMAX,
) -> np.ndarray:
    """
    Build the mel filterbank matrix M of shape (n_mels, N_BINS).

    The mel filterbank GEMM is:
        Mel[n_mels, T] = M[n_mels, N_BINS] @ P[N_BINS, T]

    Uses the standard Slaney Auditory Toolbox triangle filters with
    energy normalization, matching librosa's default mel filterbank.

    Returns:
        M: float32 array of shape (N_MELS, N_BINS) = (40, 257)
    """
    n_bins = int(1 + n_fft // 2)
    weights = np.zeros((n_mels, n_bins), dtype=np.float32)
    fftfreqs = np.linspace(0, float(sr) / 2, n_bins)

    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, n_mels + 2)
    mel_f = _mel_to_hz(mels)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    # Slaney area normalization
    enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
    weights *= enorm[:, np.newaxis]

    return weights.astype(np.float32)


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
    if sp_fft is not None:
        mfcc_all = sp_fft.dct(log_mel, type=2, norm="ortho", axis=0)
    else:
        N = log_mel.shape[0]
        k = np.arange(n_mfcc)[:, np.newaxis]
        n = np.arange(N)[np.newaxis, :]
        dct_matrix = np.cos(np.pi * k * (2 * n + 1) / (2 * N))
        dct_matrix[0] *= np.sqrt(1 / (4 * N))
        dct_matrix[1:] *= np.sqrt(1 / (2 * N))
        mfcc_all = 2.0 * (dct_matrix @ log_mel)
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


def extract_log_mel_hls(
    wav: np.ndarray,
    apply_pre_emphasis: bool = True,
    return_mfcc: bool = False,
) -> Tuple[np.ndarray, float, bool]:
    """
    Config C Preprocessing: Offloads Mel Filterbank GEMM to Custom HLS IP Core.

    Returns:
        features: float32 array (40, 101)
        hls_latency_ms: Execution time on FPGA PL
        is_hw: Boolean flag indicating physical FPGA execution
    """
    from board.app.hls_mel_runner import HLSMelRunner
    if apply_pre_emphasis:
        wav = pre_emphasis(wav)
    frames = frame_signal(wav)
    power = power_spectrum(frames)
    runner = HLSMelRunner.get_instance()
    mel_spec, hls_latency_ms, is_hw = runner.infer_gemm(power)
    log_mel = log_compression(mel_spec)
    if return_mfcc:
        return dct_mfcc(log_mel), hls_latency_ms, is_hw
    return log_mel, hls_latency_ms, is_hw


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

    # When center=False, librosa requires N_FFT samples per frame.
    # Pad right so librosa produces exactly NUM_FRAMES frames.
    needed_len = (NUM_FRAMES - 1) * HOP_LEN + N_FFT
    if len(wav) < needed_len:
        wav = np.pad(wav, (0, needed_len - len(wav)))

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

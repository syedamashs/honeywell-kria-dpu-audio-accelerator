"""
pipeline/audio_stream.py
------------------------
Unified Audio Ingestion Module supporting two modes:
  1. Passive Data Mode: Stored audio dataset files (Google Speech Commands v2 WAVs)
  2. Real-Time Voice Mode: Live microphone stream capture

Both modes output the EXACT SAME standardized audio data representation:
  - 16 kHz sample rate, single channel (mono)
  - 1.0 second duration (exactly 16,000 samples)
  - float32 NumPy array normalized to [-1.0, 1.0]

This guarantees that both passive dataset evaluation and live real-time voice
feed through the identical preprocessing and hardware inference pipeline (Configs A, B, C).
"""

from __future__ import annotations

import collections
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from pipeline.preprocessing import load_wav
from pipeline.utils import CLIP_DURATION, SAMPLE_RATE, pad_or_trim


class PassiveAudioLoader:
    """
    Handles Passive Data Mode:
    Loads stored WAV files from dataset or test evaluation subsets.
    """

    @staticmethod
    def load(path: str | Path) -> np.ndarray:
        """
        Load stored audio file, ensure 16 kHz and exactly 16,000 samples.
        Returns:
            audio: (16000,) float32 array in [-1.0, 1.0]
        """
        return load_wav(path)


class RealTimeAudioCapture:
    """
    Handles Real-Time Voice Mode:
    Captures voice continuously from microphone and maintains a rolling
    1-second audio window (16,000 samples).
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, window_duration_s: float = CLIP_DURATION):
        self.sample_rate = sample_rate
        self.window_samples = int(sample_rate * window_duration_s)
        self.buffer = collections.deque(maxlen=self.window_samples)
        # Pre-fill buffer with silence
        for _ in range(self.window_samples):
            self.buffer.append(0.0)

        self._has_sounddevice = False
        try:
            import sounddevice as sd
            self._sd = sd
            self._has_sounddevice = True
        except ImportError:
            self._sd = None

    def record_clip(self, duration_s: float = 1.0) -> np.ndarray:
        """
        Record a fixed 1-second audio clip directly from microphone.
        Returns:
            audio: (16000,) float32 array in [-1.0, 1.0]
        """
        num_samples = int(self.sample_rate * duration_s)

        if not self._has_sounddevice:
            print("[!] sounddevice not available. Generating simulated 1s ambient voice.")
            return self._generate_simulated_voice()

        try:
            print(f"[*] Recording {duration_s:.1f}s live audio from microphone... (Speak now!)")
            rec = self._sd.rec(
                frames=num_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocking=True,
            )
            audio = rec.flatten()
            print("[*] Recording captured successfully.")
            return pad_or_trim(audio, target_samples=self.window_samples)
        except Exception as e:
            print(f"[!] Warning: Microphone capture failed ({e}). Falling back to simulated input.")
            return self._generate_simulated_voice()

    def _generate_simulated_voice(self) -> np.ndarray:
        """Fallback simulated signal if physical mic is disconnected or permission denied."""
        t = np.linspace(0, 1.0, self.window_samples, endpoint=False, dtype=np.float32)
        # 440 Hz tone + harmonics + speech envelope
        audio = 0.5 * np.sin(2 * np.pi * 320 * t) + 0.3 * np.sin(2 * np.pi * 640 * t)
        env = np.sin(np.pi * np.linspace(0, 1, self.window_samples)) ** 2
        audio = (audio * env + np.random.normal(0, 0.005, size=self.window_samples)).astype(np.float32)
        return pad_or_trim(audio, self.window_samples)


def get_audio_input(
    mode: str,
    wav_path: Optional[str | Path] = None,
    capture_duration_s: float = 1.0,
) -> Tuple[np.ndarray, str]:
    """
    Unified entry point for both Passive Data Mode and Real-Time Voice Mode.

    Args:
        mode: 'passive' or 'realtime'
        wav_path: File path (required if mode == 'passive')
        capture_duration_s: Microphone capture duration (if mode == 'realtime')

    Returns:
        (audio_data, source_description)
        audio_data is guaranteed to be float32 shape (16000,) in [-1.0, 1.0]
    """
    if mode.lower() in ["passive", "file", "dataset"]:
        if not wav_path:
            raise ValueError("wav_path must be provided for Passive Data Mode.")
        p = Path(wav_path)
        audio = PassiveAudioLoader.load(p)
        desc = f"Passive Dataset File: {p.name}"
        return audio, desc

    elif mode.lower() in ["realtime", "voice", "mic", "microphone"]:
        cap = RealTimeAudioCapture()
        audio = cap.record_clip(duration_s=capture_duration_s)
        desc = "Real-Time Microphone Stream (Live Voice)"
        return audio, desc

    else:
        raise ValueError(f"Unknown input mode: '{mode}'. Must be 'passive' or 'realtime'.")

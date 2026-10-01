"""
pipeline/tests/test_audio_stream.py
------------------------------------
Unit tests for unified audio ingestion module:
  - PassiveAudioLoader
  - RealTimeAudioCapture
  - get_audio_input unified function
  - Seamless feeding into extract_log_mel preprocessing
"""

from pathlib import Path
import numpy as np
import pytest

from pipeline.audio_stream import (
    PassiveAudioLoader,
    RealTimeAudioCapture,
    get_audio_input,
)
from pipeline.preprocessing import extract_log_mel
from pipeline.utils import SAMPLE_RATE

TEST_WAV = Path(__file__).resolve().parent.parent.parent / "data" / "test_inputs" / "test_00_yes.wav"


class TestPassiveAudioLoader:
    def test_load_existing_wav(self):
        if not TEST_WAV.exists():
            pytest.skip(f"Test audio {TEST_WAV} not found")
        audio = PassiveAudioLoader.load(TEST_WAV)
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (SAMPLE_RATE,)
        assert audio.dtype == np.float32
        assert np.max(np.abs(audio)) <= 1.0

    def test_load_nonexistent_raises(self):
        with pytest.raises(Exception):
            PassiveAudioLoader.load("nonexistent_file.wav")


class TestRealTimeAudioCapture:
    def test_record_clip_shape_and_dtype(self):
        capture = RealTimeAudioCapture(sample_rate=SAMPLE_RATE, window_duration_s=1.0)
        # Even if sounddevice has no hardware mic connected, fallback produces valid simulated 1s buffer
        audio = capture.record_clip(duration_s=1.0)
        assert isinstance(audio, np.ndarray)
        assert audio.shape == (SAMPLE_RATE,)
        assert audio.dtype == np.float32
        assert not np.isnan(audio).any()
        assert np.max(np.abs(audio)) <= 1.0

    def test_simulated_voice_generation(self):
        capture = RealTimeAudioCapture(sample_rate=SAMPLE_RATE, window_duration_s=1.0)
        sim = capture._generate_simulated_voice()
        assert sim.shape == (SAMPLE_RATE,)
        assert sim.dtype == np.float32
        assert np.max(np.abs(sim)) <= 1.0
        assert not np.all(sim == 0.0)


class TestUnifiedGetAudioInput:
    def test_passive_mode(self):
        if not TEST_WAV.exists():
            pytest.skip(f"Test audio {TEST_WAV} not found")
        audio, desc = get_audio_input(mode="passive", wav_path=TEST_WAV)
        assert audio.shape == (16000,)
        assert audio.dtype == np.float32
        assert "Passive Dataset File" in desc

    def test_realtime_mode(self):
        audio, desc = get_audio_input(mode="realtime", capture_duration_s=1.0)
        assert audio.shape == (16000,)
        assert audio.dtype == np.float32
        assert "Real-Time Microphone" in desc

    def test_missing_wav_path_raises(self):
        with pytest.raises(ValueError, match="wav_path must be provided"):
            get_audio_input(mode="passive")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown input mode"):
            get_audio_input(mode="unknown_mode")


class TestPipelineEquivalence:
    """Verifies that audio from both modes feeds into the exact same preprocessing output shape."""

    def test_identical_preprocessing_tensor_shape(self):
        if not TEST_WAV.exists():
            pytest.skip(f"Test audio {TEST_WAV} not found")

        audio_passive, _ = get_audio_input(mode="passive", wav_path=TEST_WAV)
        audio_realtime, _ = get_audio_input(mode="realtime")

        feat_passive = extract_log_mel(audio_passive, apply_pre_emphasis=True)
        feat_realtime = extract_log_mel(audio_realtime, apply_pre_emphasis=True)

        assert feat_passive.shape == (40, 98)
        assert feat_realtime.shape == (40, 98)
        assert feat_passive.dtype == np.float32
        assert feat_realtime.dtype == np.float32
        assert not np.isnan(feat_passive).any()
        assert not np.isnan(feat_realtime).any()

        # Both convert to identical model input tensor [1, 1, 40, 98]
        tensor_passive = feat_passive[np.newaxis, np.newaxis, :, :]
        tensor_realtime = feat_realtime[np.newaxis, np.newaxis, :, :]
        assert tensor_passive.shape == (1, 1, 40, 98)
        assert tensor_realtime.shape == (1, 1, 40, 98)

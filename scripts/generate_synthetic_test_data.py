"""
scripts/generate_synthetic_test_data.py
---------------------------------------
Generates deterministic synthetic audio files for quick local verification
and testing of the KWS pipeline before/alongside the full Google Speech
Commands v2 download.

Generates:
  - 10 fixed 1-second 16 kHz WAV files in data/test_inputs/
  - data/test_inputs/test_manifest.json with SHA-256 hashes
  - Minimal synthetic data/processed/splits/ (train.txt, val.txt, test.txt)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav_io

from pipeline.utils import (
    GLOBAL_SEED,
    KEYWORDS,
    LABEL2IDX,
    SAMPLE_RATE,
    set_seed,
)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "data" / "test_inputs"
SPLITS_DIR = ROOT / "data" / "processed" / "splits"
SYNTH_DIR = ROOT / "data" / "synthetic"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_clip(
    label: str,
    index: int,
    seed: int = GLOBAL_SEED,
) -> np.ndarray:
    """Generate a reproducible synthetic 16 kHz audio clip."""
    rng = np.random.default_rng(seed + index * 101 + abs(hash(label)) % 10007)
    t = np.linspace(0, 1.0, SAMPLE_RATE, endpoint=False, dtype=np.float32)

    if label == "silence":
        # Low amplitude room noise
        audio = rng.normal(0, 0.005, size=SAMPLE_RATE).astype(np.float32)
    else:
        # Formant-like harmonic signal with amplitude envelope
        base_freq = 200.0 + (abs(hash(label)) % 5) * 60.0
        harmonics = [1.0, 0.6, 0.3, 0.15]
        audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        for h_idx, amp in enumerate(harmonics, start=1):
            freq = base_freq * h_idx
            audio += amp * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

        # Smooth envelope (windowing)
        env = np.sin(np.pi * np.linspace(0, 1, SAMPLE_RATE)) ** 2
        audio = audio * env * 0.7
        # Add light background noise
        audio += rng.normal(0, 0.01, size=SAMPLE_RATE).astype(np.float32)

    # Normalize to [-0.9, 0.9]
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = 0.9 * (audio / max_val)

    return (audio * 32767).astype(np.int16)


def main():
    set_seed(GLOBAL_SEED)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)

    test_labels = [
        "yes", "yes",
        "no", "no",
        "stop", "stop",
        "go", "go",
        "unknown",
        "silence",
    ]

    manifest = {}
    test_entries = []

    print("[*] Generating 10 fixed evaluation test clips...")
    for idx, label in enumerate(test_labels):
        clip_name = f"test_{idx:02d}_{label}.wav"
        out_path = TEST_DIR / clip_name
        audio_i16 = generate_synthetic_clip(label, idx)
        wav_io.write(str(out_path), SAMPLE_RATE, audio_i16)

        file_hash = sha256_file(out_path)
        manifest[f"test_{idx:02d}"] = {
            "filename": clip_name,
            "path": str(out_path),
            "label": label,
            "class_idx": LABEL2IDX[label],
            "sha256": file_hash,
        }
        test_entries.append((str(out_path), label, LABEL2IDX[label]))
        print(f"  [{idx}] {clip_name} -> label: {label:<8} sha256: {file_hash[:12]}...")

    manifest_path = TEST_DIR / "test_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest saved -> {manifest_path}")

    # Generate synthetic train/val/test splits for immediate pipeline validation
    train_entries = []
    val_entries = []
    all_classes = KEYWORDS + ["unknown", "silence"]

    print("[*] Generating synthetic train/val set for quick testing...")
    for c_idx, cls_name in enumerate(all_classes):
        cls_dir = SYNTH_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)
        # 10 train clips per class, 2 val clips per class
        for i in range(10):
            p = cls_dir / f"train_{i:02d}.wav"
            wav_io.write(str(p), SAMPLE_RATE, generate_synthetic_clip(cls_name, 100 + i))
            train_entries.append((str(p), cls_name, LABEL2IDX[cls_name]))
        for i in range(2):
            p = cls_dir / f"val_{i:02d}.wav"
            wav_io.write(str(p), SAMPLE_RATE, generate_synthetic_clip(cls_name, 200 + i))
            val_entries.append((str(p), cls_name, LABEL2IDX[cls_name]))

    for name, entries in [("train", train_entries), ("val", val_entries), ("test", test_entries)]:
        split_file = SPLITS_DIR / f"{name}.txt"
        with open(split_file, "w") as f:
            for p, lbl, c_idx in entries:
                f.write(f"{p},{lbl},{c_idx}\n")
        print(f"  Wrote {split_file} ({len(entries)} items)")

    print("[OK] Synthetic data generation complete.")


if __name__ == "__main__":
    main()

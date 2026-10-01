"""
pipeline/utils.py
-----------------
Shared utilities: reproducibility seeds, version logging, result CSV writer,
stage timer context manager.
"""

import csv
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Fixed preprocessing constants (identical across all configs) ──────────────
SAMPLE_RATE   = 16_000          # Hz
WINDOW_MS     = 25              # ms  → 400 samples
HOP_MS        = 10              # ms  → 160 samples
N_FFT         = 512
N_MELS        = 40
N_MFCC        = 40              # set equal to N_MELS; DCT step can be skipped
FMIN          = 20.0            # Hz
FMAX          = 4_000.0         # Hz
CLIP_DURATION = 1.0             # seconds
NUM_FRAMES    = 101             # floor((16000 - 400) / 160) + 1

# ── Keywords (10 + unknown + silence = 12 classes) ────────────────────────────
KEYWORDS = [
    "yes", "no", "up", "down", "left",
    "right", "on", "off", "stop", "go",
]
CLASSES  = KEYWORDS + ["unknown", "silence"]
NUM_CLASSES = len(CLASSES)                 # 12
LABEL2IDX   = {c: i for i, c in enumerate(CLASSES)}
IDX2LABEL   = {i: c for c, i in LABEL2IDX.items()}

# ── All-words in GSC v2 (35) — anything not in KEYWORDS → "unknown" ──────────
GSC_ALL_WORDS = [
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
]


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_SEED = 42


def set_seed(seed: int = GLOBAL_SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Version recording
# ─────────────────────────────────────────────────────────────────────────────

def get_versions() -> Dict[str, str]:
    """Return a dict of library name → version string."""
    libs = ["numpy", "scipy", "librosa", "torch", "torchaudio",
            "onnx", "onnxruntime", "sklearn"]
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    for lib in libs:
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "?")
        except ImportError:
            versions[lib] = "not installed"
    return versions


def print_versions() -> None:
    """Pretty-print library versions to stdout."""
    print("=" * 50)
    print("Library versions")
    print("=" * 50)
    for lib, ver in get_versions().items():
        print(f"  {lib:<20} {ver}")
    print("=" * 50)


def save_versions(path: Path) -> None:
    """Save library versions to a text file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(f"Recorded at: {datetime.now().isoformat()}\n")
        for lib, ver in get_versions().items():
            f.write(f"{lib}: {ver}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Stage timer
# ─────────────────────────────────────────────────────────────────────────────

class StageTimer:
    """Accumulates per-stage wall-clock times in nanoseconds."""

    def __init__(self):
        self._times: Dict[str, List[int]] = {}

    @contextmanager
    def measure(self, stage: str):
        t0 = time.perf_counter_ns()
        yield
        t1 = time.perf_counter_ns()
        self._times.setdefault(stage, []).append(t1 - t0)

    def stats(self, stage: str) -> Dict[str, float]:
        arr = np.array(self._times.get(stage, []), dtype=np.float64)
        if arr.size == 0:
            return {}
        return {
            "count":  int(arr.size),
            "mean_ms":   float(np.mean(arr)   / 1e6),
            "median_ms": float(np.median(arr) / 1e6),
            "p95_ms":    float(np.percentile(arr, 95) / 1e6),
            "p99_ms":    float(np.percentile(arr, 99) / 1e6),
            "min_ms":    float(np.min(arr)    / 1e6),
            "max_ms":    float(np.max(arr)    / 1e6),
        }

    def all_stats(self) -> Dict[str, Dict[str, float]]:
        return {stage: self.stats(stage) for stage in self._times}

    def print_summary(self) -> None:
        print(f"\n{'Stage':<20} {'count':>6} {'mean':>8} {'median':>8} "
              f"{'p95':>8} {'p99':>8}  (ms)")
        print("-" * 65)
        for stage, s in self.all_stats().items():
            print(f"  {stage:<18} {s['count']:>6} {s['mean_ms']:>8.2f} "
                  f"{s['median_ms']:>8.2f} {s['p95_ms']:>8.2f} {s['p99_ms']:>8.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CSV result writer
# ─────────────────────────────────────────────────────────────────────────────

RESULT_COLUMNS = [
    "board", "config", "input_id", "iteration",
    "preproc_ns", "infer_ns", "post_ns", "total_ns",
    "prediction", "correct",
]


def open_result_csv(path: Path, extra_cols: Optional[List[str]] = None):
    """Open (or append to) a CSV result file and return (file, writer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    cols = RESULT_COLUMNS + (extra_cols or [])
    fh = open(path, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    return fh, writer


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def pad_or_trim(wav: np.ndarray, target_samples: int = SAMPLE_RATE) -> np.ndarray:
    """Pad with zeros or trim to exactly target_samples."""
    if len(wav) < target_samples:
        return np.pad(wav, (0, target_samples - len(wav)))
    return wav[:target_samples]

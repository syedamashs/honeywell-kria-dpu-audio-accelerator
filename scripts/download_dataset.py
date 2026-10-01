"""
scripts/download_dataset.py
----------------------------
Step 3 — Download Google Speech Commands v2 and create train/val/test splits.

Usage:
    python scripts/download_dataset.py [--data_dir data/raw] [--select_test_inputs]

Outputs:
    data/raw/               Raw GSC v2 dataset (extracted)
    data/processed/splits/  train.txt, val.txt, test.txt  (paths + labels)
    data/test_inputs/       5-10 fixed test clips (copied + SHA-256 logged)
    data/processed/dataset_info.json

The 10-class split follows the standard convention:
    KEYWORDS (10) + 'unknown' + 'silence' = 12 classes.
Any word not in KEYWORDS → 'unknown'.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.utils import KEYWORDS, CLASSES, LABEL2IDX, set_seed, GLOBAL_SEED

# ── Dataset config ────────────────────────────────────────────────────────────
GSC_URL = (
    "https://storage.googleapis.com/download.tensorflow.org/"
    "data/speech_commands_v0.02.tar.gz"
)
GSC_ARCHIVE = "speech_commands_v0.02.tar.gz"
SILENCE_WORD = "_silence_"  # GSC v2 uses this folder for silence clips
BACKGROUND_NOISE = "_background_noise_"

# Fixed test input selection (deterministic — same clips every run)
# 2 clips per keyword × 5 keywords + 2 noise + 1 short + 1 borderline = 12 total (keep ≤10)
SELECTED_KEYWORDS_FOR_TEST = ["yes", "no", "stop", "go", "up"]
TEST_CLIPS_PER_KEYWORD = 2


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def download_dataset(data_dir: Path) -> Path:
    """Download and extract GSC v2 if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / GSC_ARCHIVE

    # Check if already extracted
    if (data_dir / "yes").exists():
        print(f"[✓] Dataset already extracted at {data_dir}")
        return data_dir

    if not archive_path.exists():
        print(f"[↓] Downloading Google Speech Commands v2 (~2.3 GB)...")
        print(f"    URL: {GSC_URL}")

        def progress(block, block_size, total):
            downloaded = block * block_size
            pct = min(100, downloaded * 100 / total) if total > 0 else 0
            mb = downloaded / 1e6
            print(f"\r    {pct:5.1f}%  {mb:6.1f} MB", end="", flush=True)

        urllib.request.urlretrieve(GSC_URL, archive_path, reporthook=progress)
        print()
        print(f"[✓] Downloaded to {archive_path}")
    else:
        print(f"[✓] Archive already exists: {archive_path}")

    print(f"[↗] Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print(f"[✓] Extracted to {data_dir}")
    return data_dir


def build_splits(
    data_dir: Path,
    splits_dir: Path,
) -> tuple[list, list, list]:
    """
    Build train/val/test file lists using GSC v2's provided split files.

    GSC v2 ships 'validation_list.txt' and 'testing_list.txt'.
    Everything not in those two files is training.

    Returns:
        train_list, val_list, test_list
        Each entry: (absolute_path, label_str, class_idx)
    """
    splits_dir.mkdir(parents=True, exist_ok=True)

    val_paths  = set((data_dir / "validation_list.txt").read_text().splitlines())
    test_paths = set((data_dir / "testing_list.txt").read_text().splitlines())

    all_words = [d.name for d in data_dir.iterdir()
                 if d.is_dir() and not d.name.startswith("_")]

    train_list, val_list, test_list = [], [], []

    for word in sorted(all_words):
        word_dir = data_dir / word
        label = word if word in KEYWORDS else "unknown"
        class_idx = LABEL2IDX[label]

        for wav in sorted(word_dir.glob("*.wav")):
            rel = f"{word}/{wav.name}"
            entry = (str(wav), label, class_idx)
            if rel in test_paths:
                test_list.append(entry)
            elif rel in val_paths:
                val_list.append(entry)
            else:
                train_list.append(entry)

    # Add silence class (generate from background noise)
    bg_dir = data_dir / BACKGROUND_NOISE
    if bg_dir.exists():
        bg_wavs = sorted(bg_dir.glob("*.wav"))
        for wav in bg_wavs:
            entry = (str(wav), "silence", LABEL2IDX["silence"])
            train_list.append(entry)

    print(f"[✓] Splits built:")
    print(f"    Train : {len(train_list):,}")
    print(f"    Val   : {len(val_list):,}")
    print(f"    Test  : {len(test_list):,}")

    # Write to text files: path,label,class_idx
    for name, lst in [("train", train_list), ("val", val_list), ("test", test_list)]:
        out = splits_dir / f"{name}.txt"
        with open(out, "w") as f:
            for path, label, idx in lst:
                f.write(f"{path},{label},{idx}\n")
        print(f"    Wrote {out}")

    return train_list, val_list, test_list


def select_test_inputs(
    data_dir: Path,
    test_dir: Path,
    test_list: list,
    seed: int = GLOBAL_SEED,
) -> dict:
    """
    Select 5-10 fixed test inputs for the official evaluation set.
    Deterministic — same clips every run.

    Returns a manifest dict: {clip_id: {path, label, sha256}}
    """
    test_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # Group test list by label
    by_label: dict[str, list] = {}
    for path, label, idx in test_list:
        by_label.setdefault(label, []).append(path)

    manifest = {}
    clip_id = 0

    for keyword in SELECTED_KEYWORDS_FOR_TEST:
        candidates = by_label.get(keyword, [])
        if not candidates:
            print(f"[!] No test clips for keyword '{keyword}', skipping")
            continue
        selected = rng.sample(candidates, min(TEST_CLIPS_PER_KEYWORD, len(candidates)))
        for src_path in selected:
            src = Path(src_path)
            dst_name = f"test_{clip_id:02d}_{keyword}_{src.name}"
            dst = test_dir / dst_name
            shutil.copy2(src, dst)
            manifest[f"test_{clip_id:02d}"] = {
                "filename": dst_name,
                "path":     str(dst),
                "label":    keyword,
                "class_idx": LABEL2IDX[keyword],
                "sha256":   sha256(dst),
            }
            clip_id += 1

    # Add one 'unknown' and one 'silence' clip
    for extra_label in ["unknown", "silence"]:
        candidates = by_label.get(extra_label, [])
        if candidates:
            src = Path(rng.choice(candidates))
            dst_name = f"test_{clip_id:02d}_{extra_label}_{src.name}"
            dst = test_dir / dst_name
            shutil.copy2(src, dst)
            manifest[f"test_{clip_id:02d}"] = {
                "filename": dst_name,
                "path":     str(dst),
                "label":    extra_label,
                "class_idx": LABEL2IDX[extra_label],
                "sha256":   sha256(dst),
            }
            clip_id += 1

    manifest_path = test_dir / "test_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[✓] Selected {len(manifest)} fixed test inputs → {test_dir}")
    print(f"    Manifest: {manifest_path}")
    return manifest


def save_dataset_info(
    data_dir: Path,
    splits_dir: Path,
    train_list, val_list, test_list,
) -> None:
    from pipeline.utils import get_versions
    info = {
        "dataset": "Google Speech Commands v2",
        "url":     GSC_URL,
        "classes": CLASSES,
        "label2idx": LABEL2IDX,
        "n_train": len(train_list),
        "n_val":   len(val_list),
        "n_test":  len(test_list),
        "versions": get_versions(),
    }
    out = splits_dir / "dataset_info.json"
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"[✓] Dataset info saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Download Google Speech Commands v2")
    parser.add_argument("--data_dir",   default="data/raw",       help="Raw data directory")
    parser.add_argument("--splits_dir", default="data/processed/splits", help="Splits output dir")
    parser.add_argument("--test_dir",   default="data/test_inputs", help="Fixed test inputs dir")
    parser.add_argument("--select_test_inputs", action="store_true",
                        help="Copy fixed test inputs to data/test_inputs/")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED)
    args = parser.parse_args()

    set_seed(args.seed)

    root = Path(__file__).parent.parent
    data_dir   = root / args.data_dir
    splits_dir = root / args.splits_dir
    test_dir   = root / args.test_dir

    # Step 1: Download
    download_dataset(data_dir)

    # Step 2: Build splits
    train_list, val_list, test_list = build_splits(data_dir, splits_dir)

    # Step 3: Save info
    save_dataset_info(data_dir, splits_dir, train_list, val_list, test_list)

    # Step 4: Fixed test inputs
    if args.select_test_inputs:
        select_test_inputs(data_dir, test_dir, test_list, seed=args.seed)

    print("\n[✓] Done. Run with --select_test_inputs to also copy fixed evaluation clips.")


if __name__ == "__main__":
    main()

"""
scripts/package_for_board.py
----------------------------
Bundles all necessary deployment files for the AMD Kria KV260 into a clean archive:
  deploy_kria_kv260.tar.gz

Includes:
  - pipeline/ (audio ingestion, preprocessing, gemm, models, utils)
  - board/ (dpu runner, demo CLI, web UI)
  - benchmarks/ (harness, cpu baseline, profiling)
  - data/test_inputs/ (curated test WAVs)
  - models/ (trained ONNX and checkpoint weights)
  - scripts/run_on_board.sh (one-shot board benchmark script)
  - requirements.txt

Excludes:
  - .git, .pytest_cache, __pycache__, scratch, large raw datasets
"""

import os
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ARCHIVE = ROOT / "deploy_kria_kv260.tar.gz"

INCLUDED_ITEMS = [
    "pipeline",
    "board",
    "benchmarks",
    "tests",
    "docs",
    "data/test_inputs",
    "models/onnx",
    "models/compiled",
    "scripts",
    "requirements.txt",
    "README.md",
    "HONEYWELL_SUBMISSION_REPORT.md",
    "HACKATHON_ENGINEERING_LOGBOOK.md",
    "hls/custom_dpu",
]


def filter_tar(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
    # Filter out cache and temp files
    name = tarinfo.name
    if "__pycache__" in name or name.endswith(".pyc") or ".pytest_cache" in name:
        return None
    return tarinfo


def make_package():
    print(f"Creating board deployment bundle: {OUTPUT_ARCHIVE.name} ...")
    with tarfile.open(OUTPUT_ARCHIVE, "w:gz") as tar:
        for item in INCLUDED_ITEMS:
            src = ROOT / item
            if src.exists():
                print(f"  + Adding {item}")
                tar.add(src, arcname=item, filter=filter_tar)
            else:
                print(f"  ! Warning: {item} not found, skipping.")

    size_mb = OUTPUT_ARCHIVE.stat().st_size / (1024 * 1024)
    print(f"\n[OK] Package created: {OUTPUT_ARCHIVE} ({size_mb:.2f} MB)")

    # Also keep deploy_kws_dpu.tar.gz in sync for HTTP server
    kws_archive = ROOT / "deploy_kws_dpu.tar.gz"
    import shutil
    shutil.copyfile(OUTPUT_ARCHIVE, kws_archive)
    print(f"[OK] Synced {kws_archive.name}")
    print("Transfer to KV260 via SCP:")
    print(f"  scp {OUTPUT_ARCHIVE.name} ubuntu@<kv260-ip>:~/")
    print("On the KV260, unpack and run:")
    print(f"  tar -xzf {OUTPUT_ARCHIVE.name}")
    print("  python3 board/app/web_ui.py")


if __name__ == "__main__":
    make_package()
